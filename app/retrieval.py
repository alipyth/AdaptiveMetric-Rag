from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import database
from .models import QueryAnalysis


TOKEN_RE = re.compile(r"[\w\u0600-\u06FF.-]+", re.UNICODE)
DATE_RE = re.compile(r"(?:\b(?:19|20)\d{2}\b|\b1[34]\d{2}\b|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
DIMENSION = 384

PERSIAN_STOP = {"از", "به", "در", "با", "برای", "که", "این", "آن", "را", "و", "یا", "چه", "چرا", "چگونه", "است", "شد", "می"}
ENGLISH_STOP = {"the", "a", "an", "of", "to", "in", "for", "is", "was", "and", "or", "what", "why", "how", "does"}


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("ي", "ی").replace("ك", "ک")
    return [t for t in TOKEN_RE.findall(normalized) if len(t) > 1 and t not in PERSIAN_STOP and t not in ENGLISH_STOP]


def best_evidence(query: str, content: str, answer: str = "") -> str:
    """Pick the sentence most responsible for a chunk matching the query."""
    query_terms = set(tokenize(query))
    answer_terms = set(tokenize(re.sub(r"\[\d+\]", "", answer)))
    important_terms = query_terms | answer_terms
    important_numbers = set(NUMBER_RE.findall(query + " " + answer))
    important_dates = set(DATE_RE.findall(query + " " + answer))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?؟؛])\s+|\n+", content) if part.strip()]
    if not sentences:
        return content[:320]

    def evidence_score(sentence: str) -> tuple[float, int]:
        terms = set(tokenize(sentence))
        answer_overlap = len(answer_terms & terms) / max(len(answer_terms), 1)
        query_overlap = len(query_terms & terms) / max(len(query_terms), 1)
        combined_overlap = len(important_terms & terms) / max(len(important_terms), 1)
        sentence_numbers = set(NUMBER_RE.findall(sentence))
        sentence_dates = set(DATE_RE.findall(sentence))
        number_bonus = .55 if important_numbers and important_numbers & sentence_numbers else 0
        date_bonus = .45 if important_dates and important_dates & sentence_dates else 0
        return .55 * answer_overlap + .30 * query_overlap + .15 * combined_overlap + number_bonus + date_bonus, -len(sentence)

    selected = max(sentences, key=evidence_score)
    return selected[:600]


def embed(text: str) -> list[float]:
    """Fast multilingual feature hashing; deterministic and requires no model download."""
    vector = [0.0] * DIMENSION
    tokens = tokenize(text)
    features = tokens + [f"{tokens[i]}::{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[idx] += sign * (1.0 if "::" not in feature else 0.55)
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return max(0.0, sum(x * y for x, y in zip(a, b)))


def analyze_query(query: str) -> QueryAnalysis:
    q = query.lower()
    language = "fa" if re.search(r"[\u0600-\u06FF]", query) else "en"
    numbers = NUMBER_RE.findall(query)
    temporal = DATE_RE.findall(query)
    quoted = re.findall(r'["«](.*?)["»]', query)
    tokens = tokenize(query)
    caps = re.findall(r"\b[A-Z][A-Za-z0-9_.-]+\b", query)
    entities = list(dict.fromkeys(quoted + caps + [t for t in tokens if any(c.isdigit() for c in t)]))[:8]

    temporal_words = ("when", "date", "expire", "released", "زمان", "تاریخ", "پایان", "منقضی", "منتشر")
    causal_words = ("why", "cause", "reason", "چرا", "علت", "دلیل")
    code_words = ("error", "exception", "stack", "cuda", "api", "function", "خطا", "کد")
    numeric_words = ("how much", "price", "dose", "count", "قیمت", "دوز", "مقدار", "چقدر")
    conceptual_words = ("how does", "explain", "concept", "topic", "subject", "summary",
                        "چگونه", "مفهوم", "توضیح", "موضوع", "درباره", "خلاصه", "چی هست", "چیه")

    if temporal or any(w in q for w in temporal_words):
        intent = "temporal_fact"
        weights = {"dense": .18, "bm25": .12, "entity": .24, "numeric": .08, "temporal": .31, "metadata": .07}
    elif numbers or any(w in q for w in numeric_words):
        intent = "numeric_fact"
        weights = {"dense": .17, "bm25": .13, "entity": .24, "numeric": .31, "temporal": .07, "metadata": .08}
    elif any(w in q for w in causal_words):
        intent = "causal"
        weights = {"dense": .46, "bm25": .18, "entity": .17, "numeric": .04, "temporal": .06, "metadata": .09}
    elif any(w in q for w in code_words):
        intent = "technical_code"
        weights = {"dense": .30, "bm25": .31, "entity": .23, "numeric": .06, "temporal": .03, "metadata": .07}
    elif any(w in q for w in conceptual_words):
        intent = "conceptual"
        weights = {"dense": .58, "bm25": .16, "entity": .08, "numeric": .03, "temporal": .03, "metadata": .12}
    else:
        intent = "exact_fact"
        weights = {"dense": .28, "bm25": .22, "entity": .26, "numeric": .08, "temporal": .07, "metadata": .09}
    return QueryAnalysis(intent=intent, language=language, weights=weights, entities=entities, numbers=numbers, temporal_terms=temporal)


def _bm25(query_tokens: list[str], doc_tokens: list[str], avg_len: float, doc_freq: Counter[str], total: int) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    counts = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = counts[term]
        if not tf:
            continue
        idf = math.log(1 + (total - doc_freq[term] + .5) / (doc_freq[term] + .5))
        denom = tf + 1.5 * (1 - .75 + .75 * len(doc_tokens) / max(avg_len, 1))
        score += idf * (tf * 2.5 / denom)
    return score


def _overlap(needles: list[str], haystack: str) -> float:
    if not needles:
        return 0.0
    low = haystack.lower()
    return sum(1 for item in needles if item.lower() in low) / len(needles)


def _numeric_signal(numbers: list[str], content: str, numeric_intent: bool) -> float:
    if numbers:
        return _overlap(numbers, content)
    return 1.0 if numeric_intent and NUMBER_RE.search(content) else 0.0


def _temporal_signal(terms: list[str], content: str, temporal_intent: bool) -> float:
    if terms:
        return _overlap(terms, content)
    return 1.0 if temporal_intent and DATE_RE.search(content) else 0.0


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    analysis: QueryAnalysis
    confidence: float
    early_exit: bool


def retrieve(query: str, candidate_count: int = 100, context_count: int = 5, filters: dict[str, Any] | None = None,
             query_vector: list[float] | None = None) -> RetrievalResult:
    analysis = analyze_query(query)
    all_chunks = database.rows(
        "SELECT c.*, d.name document_name, d.type document_type FROM chunks c JOIN documents d ON d.id=c.document_id"
    )
    if filters and filters.get("document_id"):
        all_chunks = [c for c in all_chunks if c["document_id"] == filters["document_id"]]
    if not all_chunks:
        return RetrievalResult([], analysis, 0.0, False)

    qvec, qtokens = query_vector or embed(query), tokenize(query)
    prepared: list[tuple[dict[str, Any], list[str], float]] = []
    for chunk in all_chunks:
        vector = json.loads(chunk["embedding"])
        prepared.append((chunk, json.loads(chunk["tokens"]), cosine(qvec, vector)))
    # First-stage hybrid retrieval: union strong dense and lexical candidates.
    # This exact scan is intentionally replaceable by FAISS/Qdrant at large scale.
    avg_len = sum(len(tokens) for _, tokens, _ in prepared) / len(prepared)
    doc_freq: Counter[str] = Counter()
    for _, tokens, _ in prepared:
        doc_freq.update(set(tokens))
    bm_by_id = {
        chunk["id"]: _bm25(qtokens, tokens, avg_len, doc_freq, len(prepared))
        for chunk, tokens, _ in prepared
    }
    dense_ranked = sorted(prepared, key=lambda x: x[2], reverse=True)
    lexical_ranked = sorted(prepared, key=lambda x: bm_by_id[x[0]["id"]], reverse=True)
    dense_limit = max(1, round(candidate_count * .65))
    lexical_limit = max(1, candidate_count - dense_limit)
    candidate_ids = {item[0]["id"] for item in dense_ranked[:dense_limit]}
    candidate_ids.update(item[0]["id"] for item in lexical_ranked[:lexical_limit])
    candidates = [item for item in prepared if item[0]["id"] in candidate_ids]
    bm_values = [bm_by_id[chunk["id"]] for chunk, _, _ in candidates]
    bm_max = max(bm_values, default=1) or 1

    scored: list[dict[str, Any]] = []
    for (chunk, tokens, dense_score), bm in zip(candidates, bm_values):
        content = chunk["content"]
        metadata_text = f'{chunk.get("document_name", "")} {chunk.get("section") or ""} {chunk.get("document_type", "")}'
        features = {
            "dense": dense_score,
            "bm25": bm / bm_max,
            "entity": _overlap(analysis.entities, content),
            "numeric": _numeric_signal(analysis.numbers, content, analysis.intent == "numeric_fact"),
            "temporal": _temporal_signal(analysis.temporal_terms, content, analysis.intent == "temporal_fact"),
            "metadata": _overlap(qtokens[:6], metadata_text),
        }
        final = sum(analysis.weights[key] * value for key, value in features.items())
        chunk.update(score=round(final, 6), features={k: round(v, 4) for k, v in features.items()})
        scored.append(chunk)
    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[:context_count]
    top = selected[0]["score"] if selected else 0.0
    second = selected[1]["score"] if len(selected) > 1 else 0.0
    coverage = min(1.0, sum(1 for t in set(qtokens) if t in " ".join(c["content"].lower() for c in selected)) / max(len(set(qtokens)), 1))
    calibrated_top = min(1.0, top / .62)
    separation = min(1.0, max(0.0, top - second) / .22)
    confidence = max(0.0, min(1.0, .48 * calibrated_top + .40 * coverage + .12 * separation))
    early = top > .86 and top - second > .20
    return RetrievalResult(selected, analysis, round(confidence, 3), early)
