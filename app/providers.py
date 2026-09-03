from __future__ import annotations

import re
import logging
from typing import Any

import httpx

from .models import AppSettings


logger = logging.getLogger("adaptive_metric_rag.providers")


def _has_complete_ending(answer: str) -> bool:
    clean = answer.strip()
    if not clean:
        return False
    # A citation or closing Markdown delimiter may follow the real punctuation.
    tail = re.sub(r"(?:\s*\[\d+\])+\s*$", "", clean)
    tail = re.sub(r"(?:\*\*|__|`)+\s*$", "", tail).rstrip()
    return tail.endswith((".", "!", "?", "؟", "؛", "…", ")", "]", "}", "»", '"', "'"))


def _looks_incomplete(question: str, answer: str, finish_reason: str = "") -> bool:
    clean = answer.strip()
    if not clean:
        return True
    if finish_reason.lower() in {"length", "max_tokens", "max_tokens_reached"}:
        return True
    unbalanced = clean.count("(") > clean.count(")") or clean.count("**") % 2 == 1
    broken_ending = clean.endswith(("(", "[", "{", ":", "،", ",", "-", "**"))
    multi_part = question.count("؟") + question.count("?") >= 2
    missing_sentence_end = not _has_complete_ending(clean)
    return unbalanced or broken_ending or missing_sentence_end or (multi_part and len(clean) < 350)


def _repair_instruction(language: str) -> str:
    if language == "fa":
        return ("پاسخ قبلی ناقص، قطع‌شده یا به زبان اشتباه بود. یک پاسخ کامل جایگزین بنویس؛ "
                "تمام متن توضیحی را فقط به فارسی بنویس، همه بخش‌های سؤال را جداگانه پاسخ بده، "
                "چیزی را تکرار نکن و ارجاع‌های [1]، [2] را حفظ کن. حداکثر ۱۸۰ کلمه بنویس و حتماً با یک جمله کامل تمام کن. "
                "نام خاص، ISBN و URL می‌توانند لاتین بمانند.")
    return "The previous answer was incomplete or cut off. Write a complete replacement in at most 180 words, answer every part separately, avoid repetition, preserve [1], [2] citations, and end with a complete sentence."


def _continuation_instruction(language: str) -> str:
    if language == "fa":
        return ("پاسخ دقیقاً در انتهای متن قطع شده است. فقط از همان عبارت ناتمام ادامه بده؛ متن قبلی را تکرار نکن، "
                "پاسخ را در حداکثر ۱۲۰ کلمه جمع‌بندی کن، ارجاع‌ها را حفظ کن و حتماً با جمله کامل پایان بده.")
    return ("The answer was cut off at the very end. Continue exactly from the unfinished phrase without repeating "
            "the previous text, finish within 120 words, preserve citations, and end with a complete sentence.")


def _join_continuation(answer: str, continuation: str) -> str:
    return answer.rstrip() + ("" if answer.rstrip().endswith(("-", "—")) else " ") + continuation.lstrip()


def _trim_incomplete_tail(answer: str) -> str:
    """Remove only a final dangling fragment when a provider repeatedly stops early."""
    clean = answer.strip()
    if _has_complete_ending(clean):
        return clean
    endings = list(re.finditer(r"[.!?؟؛…](?:\s*\[\d+\])*", clean))
    if endings and endings[-1].end() >= 40:
        return clean[:endings[-1].end()].rstrip()
    return clean


def _language_instruction(language: str) -> str:
    if language == "fa":
        return ("قانون الزامی زبان: پاسخ را فقط به فارسی روان بنویس. متن انگلیسی منابع را ترجمه و به فارسی توضیح بده. "
                "فقط نام‌های خاص، کد، ISBN و نشانی اینترنتی را در صورت نیاز لاتین نگه دار. این قانون بر همه دستورهای دیگر مقدم است.")
    return "Mandatory language rule: write the complete answer in English."


def _wrong_language(language: str, answer: str) -> bool:
    """Detect a clearly English response to a Persian question without rejecting names/URLs."""
    if language != "fa":
        return False
    prose = re.sub(r"https?://\S+|\[[0-9]+\]|`[^`]*`", " ", answer)
    persian = len(re.findall(r"[\u0600-\u06ff]", prose))
    latin = len(re.findall(r"[A-Za-z]", prose))
    return latin >= 35 and (persian < 18 or persian < latin * .22)


def _finalize_answer(answer: str) -> str:
    """Prevent a repaired model response from ending with open formatting."""
    clean = _trim_incomplete_tail(answer)
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


def build_prompt(question: str, chunks: list[dict[str, Any]], system_prompt: str, language: str) -> tuple[str, str]:
    sources = "\n\n".join(
        f"[{i}] Document: {chunk['document_name']} | Page: {chunk.get('page') or '-'}\n{chunk['content']}"
        for i, chunk in enumerate(chunks, 1)
    )
    if language == "fa":
        user = (f"سؤال:\n{question}\n\nمنابع:\n{sources}\n\n"
                "پاسخی کامل، مستقیم و مستند به همین منابع ارائه کن. تمام بخش‌های سؤال را پاسخ بده، "
                "ارجاع درون‌متنی [1]، [2] درج کن و جمله را نیمه‌تمام رها نکن. پاسخ باید فارسی باشد.")
    else:
        user = f"Question:\n{question}\n\nSources:\n{sources}\n\nGive a complete, direct answer grounded in these sources. Answer every requested part separately, include inline citations, and do not stop mid-sentence."
    return system_prompt.strip() + "\n\n" + _language_instruction(language), user


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
    system, user = build_prompt(question, chunks, settings.system_prompt, language)
    timeout = httpx.Timeout(150.0, connect=15.0)
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
            done_reason = payload.get("done_reason", "")
            logger.warning("Ollama generation finished: reason=%s chars=%d eval_count=%s complete=%s", done_reason, len(answer), payload.get("eval_count"), not _looks_incomplete(question, answer, done_reason))
            if _wrong_language(language, answer):
                repair = await client.post(f"{base}/api/chat", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer},
                                 {"role": "user", "content": _repair_instruction(language)}],
                    "options": {"temperature": min(settings.temperature, .2), "num_predict": settings.max_tokens},
                })
                repair.raise_for_status()
                repaired_payload = repair.json()
                answer = repaired_payload["message"]["content"]
                done_reason = repaired_payload.get("done_reason", "")
                logger.warning("Ollama language repair finished: reason=%s chars=%d eval_count=%s complete=%s", done_reason, len(answer), repaired_payload.get("eval_count"), not _looks_incomplete(question, answer, done_reason))
            for attempt in range(2):
                if not _looks_incomplete(question, answer, done_reason):
                    break
                continuation = await client.post(f"{base}/api/chat", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer},
                                 {"role": "user", "content": _continuation_instruction(language)}],
                    "options": {"temperature": min(settings.temperature, .2), "num_predict": min(settings.max_tokens, 512)},
                })
                continuation.raise_for_status()
                continuation_payload = continuation.json()
                piece = continuation_payload["message"]["content"]
                answer = _join_continuation(answer, piece)
                done_reason = continuation_payload.get("done_reason", "")
                logger.warning("Ollama continuation %d finished: reason=%s piece_chars=%d total_chars=%d complete=%s", attempt + 1, done_reason, len(piece), len(answer), not _looks_incomplete(question, answer, done_reason))
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
            if _looks_incomplete(question, answer, choice.get("finish_reason", "")) or _wrong_language(language, answer):
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
            if _looks_incomplete(question, answer, candidate.get("finishReason", "")) or _wrong_language(language, answer):
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
