from app.documents import chunk_blocks
from app.retrieval import analyze_query, best_evidence, cosine, embed, expanded_keywords, normalize_query, query_variants


def test_temporal_router_prioritizes_time():
    result = analyze_query("قرارداد شماره 137 چه زمانی تمام می‌شود؟")
    assert result.intent == "temporal_fact"
    assert result.weights["temporal"] > result.weights["dense"]
    assert "137" in result.numbers


def test_causal_router_prioritizes_semantics():
    result = analyze_query("چرا این قرارداد فسخ شده؟")
    assert result.intent == "causal"
    assert result.weights["dense"] == max(result.weights.values())


def test_book_topic_query_is_conceptual():
    result = analyze_query("موضوع کتاب چیه؟")
    assert result.intent == "conceptual"
    assert result.weights["dense"] == max(result.weights.values())


def test_document_browse_intent_and_multilingual_keywords():
    result = analyze_query("از متن کتاب برام بنویس")
    assert result.intent == "document_browse"
    assert "book" in result.keywords
    assert "text" in result.keywords
    assert len(query_variants("از متن کتاب برام بنویس")) > 1


def test_conversational_creativity_queries_normalize_consistently():
    assert normalize_query("خوب چطوری خلاق باشیم؟") == "چطور خلاق باشیم؟"
    first = analyze_query("خوب چطوری خلاق باشیم؟")
    second = analyze_query("چطور خلاق باشیم؟")
    assert first.intent == second.intent == "conceptual"
    assert first.keywords == second.keywords
    assert "creativity" in first.keywords


def test_embedding_is_deterministic_and_normalized():
    first = embed("CUDA out of memory")
    second = embed("CUDA out of memory")
    assert first == second
    assert abs(cosine(first, first) - 1) < 1e-6


def test_chunk_overlap():
    text = "A sentence. " * 300
    chunks = chunk_blocks([(text, 1, "Intro")], 300, 50)
    assert len(chunks) > 2
    assert all(chunk["page"] == 1 for chunk in chunks)


def test_evidence_uses_final_answer_to_pick_supporting_sentence():
    source = "قرارداد شماره ۱۳۷ در فروردین امضا شد. مبلغ کل قرارداد ۲۴۰ میلیون ریال است. قرارداد یک سال اعتبار دارد."
    evidence = best_evidence(
        "هزینه قرارداد چقدر است؟",
        source,
        "مبلغ قرارداد ۲۴۰ میلیون ریال است [1].",
    )
    assert "۲۴۰ میلیون ریال" in evidence
