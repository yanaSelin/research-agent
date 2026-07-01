import json

from backends.local import LocalCorpusBackend  # type: ignore[import-not-found]


def _write_corpus(tmp_path):
    corpus = {
        "documents": [
            {"id": "d1", "url": "https://encyclopedia.example/eiffel",
             "domain": "encyclopedia.example", "published": "2019-04-11",
             "title": "Eiffel Tower", "content": "The Eiffel Tower was completed in 1889."},
            {"id": "d2", "url": "https://forum.example/t/1", "domain": "forum.example",
             "published": "2021-06-14", "title": "eiffel thread",
             "content": "I heard the tower was built in 1887 actually."},
            {"id": "d3", "url": "https://news.reuters.example/moon", "domain": "news.reuters.example",
             "published": "2020-01-01", "title": "Moon landing",
             "content": "Apollo 11 landed on the moon in 1969."},
        ]
    }
    p = tmp_path / "web_corpus.json"
    p.write_text(json.dumps(corpus), encoding="utf-8")
    return p


def test_returns_only_query_relevant_hits(tmp_path):
    backend = LocalCorpusBackend(_write_corpus(tmp_path))
    hits = backend.search("When was the Eiffel Tower completed?", k=6)
    domains = {h.domain for h in hits}
    assert "encyclopedia.example" in domains
    assert "news.reuters.example" not in domains  # unrelated, score 0
    top = hits[0]
    assert top.url == "https://encyclopedia.example/eiffel"
    assert top.published == "2019-04-11"


def test_respects_k(tmp_path):
    backend = LocalCorpusBackend(_write_corpus(tmp_path))
    hits = backend.search("eiffel tower", k=1)
    assert len(hits) == 1
