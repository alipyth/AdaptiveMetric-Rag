from __future__ import annotations

import re
from typing import Any

import httpx

from .models import AppSettings


def _looks_incomplete(question: str, answer: str, finish_reason: str = "") -> bool:
    clean = answer.strip()
    if not clean:
        return True
    if finish_reason.lower() in {"length", "max_tokens", "max_tokens_reached"}:
        return True
    unbalanced = clean.count("(") > clean.count(")") or clean.count("**") % 2 == 1
    broken_ending = clean.endswith(("(", "[", "{", ":", "،", ",", "-", "**"))
    multi_part = question.count("؟") + question.count("?") >= 2
    return unbalanced or broken_ending or (multi_part and len(clean) < 350)


def _repair_instruction(language: str) -> str:
    if language == "fa":
        return "پاسخ قبلی ناقص یا قطع‌شده بود. یک پاسخ کامل جایگزین بنویس؛ همه بخش‌های سؤال را جداگانه پاسخ بده، چیزی را تکرار نکن و citationهای [1]، [2] را حفظ کن."
    return "The previous answer was incomplete or cut off. Write a complete replacement, answer every part separately, avoid repetition, and preserve [1], [2] citations."


def _finalize_answer(answer: str) -> str:
    """Prevent a repaired model response from ending with open formatting."""
    clean = answer.strip()
    if clean.count("**") % 2 == 1:
        clean += "**"
    missing_parentheses = clean.count("(") - clean.count(")")
    if 0 < missing_parentheses <= 3:
        clean += ")" * missing_parentheses
    missing_brackets = clean.count("[") - clean.count("]")
    if 0 < missing_brackets <= 3:
        clean += "]" * missing_brackets
    if clean.endswith((":", "،", ",", "-")):
        clean = clean.rstrip(":،,- ") + "."
    return clean


def build_prompt(question: str, chunks: list[dict[str, Any]], system_prompt: str) -> tuple[str, str]:
    sources = "\n\n".join(
        f"[{i}] Document: {chunk['document_name']} | Page: {chunk.get('page') or '-'}\n{chunk['content']}"
        for i, chunk in enumerate(chunks, 1)
    )
    user = f"Question:\n{question}\n\nSources:\n{sources}\n\nGive a complete, direct answer grounded in these sources. Answer every requested part separately, include inline citations, and do not stop mid-sentence."
    return system_prompt, user


def local_answer(question: str, chunks: list[dict[str, Any]], language: str) -> str:
    if not chunks:
        return "سند مرتبطی پیدا نشد. لطفاً منبع مناسب اضافه کنید." if language == "fa" else "No relevant source was found. Please add a suitable document."
    excerpts = []
    for i, chunk in enumerate(chunks[:3], 1):
        sentences = re.split(r"(?<=[.!?؟])\s+|\n+", chunk["content"])
        excerpt = next((s.strip() for s in sentences if len(s.strip()) > 35), chunk["content"][:320]).strip()
        excerpts.append(f"{excerpt} [{i}]")
    prefix = "بر اساس منابع بازیابی‌شده:" if language == "fa" else "Based on the retrieved sources:"
    return prefix + "\n\n" + "\n\n".join(excerpts)


async def generate(settings: AppSettings, question: str, chunks: list[dict[str, Any]], language: str) -> str:
    if settings.provider == "local":
        return local_answer(question, chunks, language)
    system, user = build_prompt(question, chunks, settings.system_prompt)
    timeout = httpx.Timeout(90.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if settings.provider == "ollama":
            base = (settings.base_url or "http://host.docker.internal:11434").rstrip("/")
            response = await client.post(f"{base}/api/chat", json={
                "model": settings.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": settings.temperature, "num_predict": settings.max_tokens},
            })
            response.raise_for_status()
            payload = response.json()
            answer = payload["message"]["content"]
            if _looks_incomplete(question, answer, payload.get("done_reason", "")):
                repair = await client.post(f"{base}/api/chat", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer},
                                 {"role": "user", "content": _repair_instruction(language)}],
                    "options": {"temperature": min(settings.temperature, .2), "num_predict": settings.max_tokens},
                })
                repair.raise_for_status()
                answer = repair.json()["message"]["content"]
            return _finalize_answer(answer)
        if settings.provider == "openai":
            base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
            response = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {settings.api_key}"}, json={
                "model": settings.model, "temperature": settings.temperature, "max_tokens": settings.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            answer = choice["message"]["content"]
            if _looks_incomplete(question, answer, choice.get("finish_reason", "")):
                repair = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {settings.api_key}"}, json={
                    "model": settings.model, "temperature": min(settings.temperature, .2), "max_tokens": settings.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer}, {"role": "user", "content": _repair_instruction(language)}],
                })
                repair.raise_for_status()
                answer = repair.json()["choices"][0]["message"]["content"]
            return _finalize_answer(answer)
        if settings.provider == "gemini":
            base = (settings.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            response = await client.post(f"{base}/models/{settings.model}:generateContent", params={"key": settings.api_key}, json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": settings.temperature, "maxOutputTokens": settings.max_tokens},
            })
            response.raise_for_status()
            payload = response.json()
            candidate = payload["candidates"][0]
            answer = candidate["content"]["parts"][0]["text"]
            if _looks_incomplete(question, answer, candidate.get("finishReason", "")):
                repair_prompt = user + "\n\n" + _repair_instruction(language)
                repair = await client.post(f"{base}/models/{settings.model}:generateContent", params={"key": settings.api_key}, json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": repair_prompt}]}],
                    "generationConfig": {"temperature": min(settings.temperature, .2), "maxOutputTokens": settings.max_tokens},
                })
                repair.raise_for_status()
                answer = repair.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _finalize_answer(answer)
    raise ValueError(f"Unknown provider: {settings.provider}")
