from app.providers import _finalize_answer, _join_continuation, _looks_incomplete, _wrong_language, build_prompt


def test_detects_cut_off_answer():
    assert _looks_incomplete("نویسنده کیست؟", "نویسنده سینتیا پن (")


def test_detects_plain_text_cut_off_mid_sentence():
    assert _looks_incomplete("چطور خلاق باشیم؟", "هدف این مسیر نشان دادن این واقعیت است که فارغ از شخصیت")


def test_accepts_complete_sentence_with_citation():
    assert not _looks_incomplete("چطور خلاق باشیم؟", "خلاقیت مهارتی قابل تمرین است. [2]")


def test_continuation_is_joined_and_dangling_tail_is_removed():
    joined = _join_continuation("خلاقیت مهارتی است که", "می‌توان آن را تمرین کرد. [2]")
    assert joined == "خلاقیت مهارتی است که می‌توان آن را تمرین کرد. [2]"
    assert _finalize_answer(joined + " این جمله ناتمام") == joined


def test_detects_short_multi_part_answer():
    assert _looks_incomplete("نویسنده کیست؟ ناشر چیست؟ ISBN چند است؟", "نویسنده سینتیا پن است.")


def test_accepts_complete_answer():
    answer = "نویسنده سینتیا پن است. ناشر سیتی بوکس است. شماره ISBN نیز در منبع ذکر نشده است. " * 6
    assert not _looks_incomplete("نویسنده کیست؟ ناشر چیست؟ ISBN چند است؟", answer)


def test_finalizer_closes_open_formatting():
    result = _finalize_answer("**نویسنده:** سینتیا پن (")
    assert result.endswith(")")
    assert result.count("(") == result.count(")")


def test_persian_language_is_enforced_after_custom_system_prompt():
    chunks = [{"document_name": "book.pdf", "page": 1, "content": "Creativity can be practiced."}]
    system, user = build_prompt("چطور خلاق باشیم؟", chunks, "Answer in English.", "fa")
    assert system.rfind("فقط به فارسی") > system.find("Answer in English")
    assert "پاسخ باید فارسی باشد" in user


def test_wrong_language_detection_allows_latin_names_but_rejects_english_answer():
    english = "Creativity develops through deliberate practice, curiosity, experiments, and reflection over time."
    mixed = "برای خلاق‌تر شدن، تمرین و کنجکاوی را جدی بگیرید. کتاب You Are Creative نیز به همین موضوع می‌پردازد."
    assert _wrong_language("fa", english)
    assert not _wrong_language("fa", mixed)
