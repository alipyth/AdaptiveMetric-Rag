from app.providers import _finalize_answer, _looks_incomplete


def test_detects_cut_off_answer():
    assert _looks_incomplete("نویسنده کیست؟", "نویسنده سینتیا پن (")


def test_detects_short_multi_part_answer():
    assert _looks_incomplete("نویسنده کیست؟ ناشر چیست؟ ISBN چند است؟", "نویسنده سینتیا پن است.")


def test_accepts_complete_answer():
    answer = "نویسنده سینتیا پن است. ناشر سیتی بوکس است. شماره ISBN نیز در منبع ذکر نشده است. " * 6
    assert not _looks_incomplete("نویسنده کیست؟ ناشر چیست؟ ISBN چند است؟", answer)


def test_finalizer_closes_open_formatting():
    result = _finalize_answer("**نویسنده:** سینتیا پن (")
    assert result.endswith(")")
    assert result.count("(") == result.count(")")
