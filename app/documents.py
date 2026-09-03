from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from . import database
from .embeddings import create_embeddings, document_embedding_text
from .models import AppSettings
from .retrieval import tokenize


ALLOWED = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".htm"}


def extract(filename: str, payload: bytes) -> list[tuple[str, int | None, str | None]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(payload))
        return [(page.extract_text() or "", i + 1, None) for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        doc = DocxDocument(io.BytesIO(payload))
        blocks, section = [], None
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style and paragraph.style.name.startswith("Heading"):
                section = text
            blocks.append((text, None, section))
        return blocks
    text = payload.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text("\n", strip=True)
    elif suffix == ".json":
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    elif suffix == ".csv":
        rows = csv.reader(io.StringIO(text))
        text = "\n".join(" | ".join(row) for row in rows)
    return [(text, None, None)]


def chunk_blocks(blocks: list[tuple[str, int | None, str | None]], size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for text, page, section in blocks:
        clean = re.sub(r"[ \t]+", " ", text).strip()
        if not clean:
            continue
        start = 0
        while start < len(clean):
            end = min(len(clean), start + size)
            if end < len(clean):
                boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end), clean.rfind("؟", start, end))
                if boundary > start + size // 2:
                    end = boundary + 1
            content = clean[start:end].strip()
            if content:
                chunks.append({"content": content, "page": page, "section": section})
            if end >= len(clean):
                break
            start = max(start + 1, end - overlap)
    return chunks


async def ingest(filename: str, content_type: str, payload: bytes, chunk_size: int, overlap: int,
                 settings: AppSettings) -> dict:
    doc_id = uuid.uuid4().hex
    pieces = chunk_blocks(extract(filename, payload), chunk_size, overlap)
    vectors = await create_embeddings(
        settings,
        [document_embedding_text(filename, piece.get("section"), piece["content"]) for piece in pieces],
    )
    now = datetime.now(timezone.utc).isoformat()
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
            (doc_id, filename, content_type or "application/octet-stream", len(payload), len(pieces), now, "{}"),
        )
        for position, (piece, vector) in enumerate(zip(pieces, vectors)):
            text = piece["content"]
            db.execute(
                "INSERT INTO chunks(id,document_id,position,page,section,content,embedding,tokens,metadata) VALUES(?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, doc_id, position, piece["page"], piece["section"], text,
                 database.json_value(vector), database.json_value(tokenize(text)), "{}"),
            )
    return {"id": doc_id, "name": filename, "type": content_type, "size": len(payload), "chunks": len(pieces), "created_at": now}
