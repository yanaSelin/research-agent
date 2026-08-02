from backends.base import SearchHit  # type: ignore[import-not-found]
from hitfmt import format_hits, parse_hits  # type: ignore[import-not-found]


def test_round_trip_web_hit():
    hits = [SearchHit("https://encyclopedia.example/e", "encyclopedia.example",
                      "2019-04-11", "Eiffel Tower", "The tower was completed in 1889.")]
    text = format_hits(hits)
    parsed = parse_hits(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p.id == 1
    assert p.title == "Eiffel Tower"
    assert p.domain == "encyclopedia.example"
    assert p.published == "2019-04-11"
    assert p.url == "https://encyclopedia.example/e"
    assert "completed in 1889" in p.content


def test_internal_hit_empty_url_and_null_published():
    hits = [SearchHit("", "internal.corp", None, "Audit", "vault C details here.")]
    parsed = parse_hits(format_hits(hits))
    assert parsed[0].url == ""
    assert parsed[0].published is None


def test_numbering_starts_and_increments():
    hits = [
        SearchHit("https://a.example/1", "a.example", "2020-01-01", "A", "alpha content"),
        SearchHit("https://b.example/2", "b.example", "2020-01-02", "B", "beta content"),
    ]
    parsed = parse_hits(format_hits(hits, start=1))
    assert [p.id for p in parsed] == [1, 2]


def test_parse_skips_malformed_block():
    assert parse_hits("not a hit block at all") == []


def test_url_with_pipe_is_preserved():
    hits = [SearchHit("https://news.example/a?ref=x|y", "news.example",
                      "2020-01-01", "Title", "content")]
    parsed = parse_hits(format_hits(hits))
    assert parsed[0].url == "https://news.example/a?ref=x|y"
    assert parsed[0].domain == "news.example"
    assert parsed[0].published == "2020-01-01"
