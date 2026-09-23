from fse.ingest import OVERLAP, chunk_page


def test_short_page_stays_whole():
    assert chunk_page("Net income was $1,577 million.") == ["Net income was $1,577 million."]


def test_empty_page_gives_no_chunks():
    assert chunk_page("   \n\t ") == []


def test_long_page_covers_every_character_with_overlap():
    text = "".join(chr(65 + i % 26) for i in range(5000))
    chunks = chunk_page(text, size=1200, overlap=OVERLAP)
    assert all(len(c) <= 1200 for c in chunks)
    assert chunks[0] == text[:1200]
    assert chunks[-1].endswith(text[-50:])            # the tail is not dropped
    for a, b in zip(chunks, chunks[1:]):
        assert a[-OVERLAP:] == b[:OVERLAP]              # consecutive windows overlap exactly


def test_whitespace_is_normalised():
    assert chunk_page("a\n\nb   c") == ["a b c"]
