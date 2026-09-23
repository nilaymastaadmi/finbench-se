from fse.judge import strip_confidence


def test_confidence_on_its_own_line():
    assert strip_confidence("Capex was $1,577 million.\nConfidence: 85") == ("Capex was $1,577 million.", 85)


def test_confidence_on_the_same_line():
    # the smoke test's real output shape
    text = "The excerpts do not contain this information. Confidence: 0"
    assert strip_confidence(text) == ("The excerpts do not contain this information.", 0)


def test_missing_confidence_is_none():
    assert strip_confidence("The excerpts do not contain this information.") == (
        "The excerpts do not contain this information.", None)


def test_word_confidence_inside_an_answer_is_not_a_score():
    text = "Management expressed confidence in 2019 guidance.\nConfidence: 40%"
    assert strip_confidence(text) == ("Management expressed confidence in 2019 guidance.", 40)


def test_out_of_range_confidence_is_rejected():
    assert strip_confidence("Answer.\nConfidence: 450")[1] is None
