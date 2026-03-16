# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
"""Unit tests for researchclaw.literature module.

All network-dependent tests mock HTTP responses via monkeypatch.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.literature.models import Author, Paper
from researchclaw.literature.semantic_scholar import (
    _parse_s2_paper,
    search_semantic_scholar,
)
from researchclaw.literature.arxiv_client import (
    _parse_atom_feed,
    search_arxiv,
)
from researchclaw.literature.search import (
    _deduplicate,
    _normalise_title,
    papers_to_bibtex,
    search_papers,
    search_papers_multi_query,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────


def _make_paper(**kwargs: Any) -> Paper:
    defaults = {
        "paper_id": "s2-abc",
        "title": "Attention Is All You Need",
        "authors": (Author(name="Ashish Vaswani"),),
        "year": 2017,
        "venue": "NeurIPS",
        "citation_count": 80000,
        "doi": "10.5555/3295222.3295349",
        "arxiv_id": "1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
        "source": "semantic_scholar",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


SAMPLE_S2_RESPONSE = {
    "total": 1,
    "data": [
        {
            "paperId": "abc123",
            "title": "Test Paper on Transformers",
            "abstract": "We study transformers for NLP tasks.",
            "year": 2024,
            "venue": "NeurIPS",
            "citationCount": 42,
            "authors": [
                {"authorId": "1", "name": "Jane Smith"},
                {"authorId": "2", "name": "John Doe"},
            ],
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.00001"},
            "url": "https://www.semanticscholar.org/paper/abc123",
        }
    ],
}


SAMPLE_ARXIV_ATOM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>A Novel Approach to Protein Folding</title>
        <summary>We propose a new method for protein structure prediction.</summary>
        <published>2024-01-15T00:00:00Z</published>
        <author><name>Alice Researcher</name></author>
        <author><name>Bob Scientist</name></author>
        <link href="http://arxiv.org/abs/2401.00001v1" type="text/html"/>
        <arxiv:primary_category term="cs.AI"/>
        <arxiv:doi>10.5678/protein</arxiv:doi>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2402.00002v1</id>
        <title>Deep Reinforcement Learning Survey</title>
        <summary>A comprehensive survey of deep RL methods.</summary>
        <published>2024-02-20T00:00:00Z</published>
        <author><name>Charlie Expert</name></author>
        <link href="http://arxiv.org/abs/2402.00002v1" type="text/html"/>
        <arxiv:primary_category term="cs.LG"/>
      </entry>
    </feed>
""")


# ──────────────────────────────────────────────────────────────────────
# Author tests
# ──────────────────────────────────────────────────────────────────────


class TestAuthor:
    def test_last_name_simple(self) -> None:
        a = Author(name="Jane Smith")
        assert a.last_name() == "smith"

    def test_last_name_accented(self) -> None:
        a = Author(name="José García")
        assert a.last_name() == "garcia"  # accent stripped, but 'i' preserved

    def test_last_name_single(self) -> None:
        a = Author(name="Madonna")
        assert a.last_name() == "madonna"

    def test_last_name_empty(self) -> None:
        a = Author(name="")
        assert a.last_name() == "unknown"


# ──────────────────────────────────────────────────────────────────────
# Paper tests
# ──────────────────────────────────────────────────────────────────────


class TestPaper:
    def test_cite_key_format(self) -> None:
        p = _make_paper()
        key = p.cite_key
        assert key == "vaswani2017attention"

    def test_cite_key_no_authors(self) -> None:
        p = _make_paper(authors=())
        assert p.cite_key.startswith("anon")

    def test_cite_key_no_year(self) -> None:
        p = _make_paper(year=0)
        assert "0000" in p.cite_key

    def test_to_bibtex_contains_fields(self) -> None:
        p = _make_paper()
        bib = p.to_bibtex()
        assert "@inproceedings{vaswani2017attention," in bib
        assert "title = {Attention Is All You Need}" in bib
        assert "author = {Ashish Vaswani}" in bib
        assert "year = {2017}" in bib
        assert "doi = {10.5555/3295222.3295349}" in bib
        assert "eprint = {1706.03762}" in bib

    def test_to_bibtex_override(self) -> None:
        p = _make_paper(_bibtex_override="@article{custom, title={Custom}}")
        assert p.to_bibtex() == "@article{custom, title={Custom}}"

    def test_to_bibtex_article_no_venue(self) -> None:
        p = _make_paper(venue="", arxiv_id="2301.00001")
        bib = p.to_bibtex()
        assert "@article{" in bib
        assert "journal = {arXiv preprint arXiv:2301.00001}" in bib

    def test_to_bibtex_arxiv_category_venue(self) -> None:
        """T1.4: arXiv category codes (cs.CL) must not be used as journal names."""
        p = _make_paper(venue="cs.CL", arxiv_id="2301.00001")
        bib = p.to_bibtex()
        assert "journal = {cs.CL}" not in bib
        assert "arXiv preprint" in bib

    def test_to_dict(self) -> None:
        p = _make_paper()
        d = p.to_dict()
        assert d["paper_id"] == "s2-abc"
        assert d["cite_key"] == "vaswani2017attention"
        assert isinstance(d["authors"], list)
        assert d["authors"][0]["name"] == "Ashish Vaswani"

    def test_paper_frozen(self) -> None:
        p = _make_paper()
        with pytest.raises(AttributeError):
            p.title = "new title"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Semantic Scholar client tests
# ──────────────────────────────────────────────────────────────────────


class TestSemanticScholar:
    def test_parse_s2_paper(self) -> None:
        item = SAMPLE_S2_RESPONSE["data"][0]
        p = _parse_s2_paper(item)
        assert p.paper_id == "s2-abc123"
        assert p.title == "Test Paper on Transformers"
        assert len(p.authors) == 2
        assert p.authors[0].name == "Jane Smith"
        assert p.year == 2024
        assert p.doi == "10.1234/test"
        assert p.arxiv_id == "2401.00001"
        assert p.source == "semantic_scholar"
        assert p.citation_count == 42

    def test_search_semantic_scholar_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock urllib to return sample S2 response."""
        # Reset S2 circuit breaker (may be tripped from prior test API calls)
        from researchclaw.literature.semantic_scholar import _reset_circuit_breaker
        _reset_circuit_breaker()

        response_bytes = json.dumps(SAMPLE_S2_RESPONSE).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.urllib.request.urlopen",
            lambda *a, **kw: mock_resp,
        )

        papers = search_semantic_scholar("transformers", limit=5)
        assert len(papers) == 1
        assert papers[0].title == "Test Paper on Transformers"

    def test_search_semantic_scholar_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return empty list on network error."""
        from researchclaw.literature.semantic_scholar import _reset_circuit_breaker
        _reset_circuit_breaker()

        import urllib.error

        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.urllib.request.urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("timeout")),
        )
        # Patch sleep to speed up test
        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.time.sleep", lambda _: None
        )
        papers = search_semantic_scholar("test", limit=5)
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# arXiv client tests
# ──────────────────────────────────────────────────────────────────────


class TestArxiv:
    def test_parse_atom_feed(self) -> None:
        papers = _parse_atom_feed(SAMPLE_ARXIV_ATOM)
        assert len(papers) == 2

        p1 = papers[0]
        assert p1.title == "A Novel Approach to Protein Folding"
        assert p1.arxiv_id == "2401.00001"
        assert p1.year == 2024
        assert len(p1.authors) == 2
        assert p1.authors[0].name == "Alice Researcher"
        assert p1.source == "arxiv"
        assert p1.doi == "10.5678/protein"

        p2 = papers[1]
        assert p2.title == "Deep Reinforcement Learning Survey"
        assert p2.arxiv_id == "2402.00002"
        assert p2.doi == ""  # no doi

    def test_search_arxiv_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_ARXIV_ATOM.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client.urllib.request.urlopen",
            lambda *a, **kw: mock_resp,
        )

        papers = search_arxiv("protein folding", limit=10)
        assert len(papers) == 2

    def test_parse_atom_feed_empty(self) -> None:
        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        papers = _parse_atom_feed(xml)
        assert papers == []

    def test_parse_atom_feed_invalid_xml(self) -> None:
        papers = _parse_atom_feed("not xml at all <><>")
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# Unified search & deduplication tests
# ──────────────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_dedup_by_doi(self) -> None:
        p1 = _make_paper(paper_id="s2-1", doi="10.1234/a", citation_count=100)
        p2 = _make_paper(
            paper_id="arxiv-1", doi="10.1234/a", citation_count=50, source="arxiv"
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 100  # keeps higher

    def test_dedup_by_arxiv_id(self) -> None:
        p1 = _make_paper(
            paper_id="s2-1", doi="", arxiv_id="2401.00001", citation_count=10
        )
        p2 = _make_paper(
            paper_id="arxiv-1",
            doi="",
            arxiv_id="2401.00001",
            citation_count=20,
            source="arxiv",
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 20  # arxiv version had more

    def test_dedup_by_title(self) -> None:
        p1 = _make_paper(
            paper_id="s2-1",
            doi="",
            arxiv_id="",
            title="My Cool Paper",
            citation_count=5,
        )
        p2 = _make_paper(
            paper_id="s2-2",
            doi="",
            arxiv_id="",
            title="My Cool Paper",
            citation_count=10,
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 10

    def test_dedup_no_duplicates(self) -> None:
        p1 = _make_paper(paper_id="s2-1", title="Paper A", doi="10.1/a", arxiv_id="1111.11111")
        p2 = _make_paper(paper_id="s2-2", title="Paper B", doi="10.1/b", arxiv_id="2222.22222")
        result = _deduplicate([p1, p2])
        assert len(result) == 2

    def test_normalise_title(self) -> None:
        assert _normalise_title("  The Great Paper!!! ") == "the great paper"
        assert _normalise_title("A/B Testing: Methods") == "ab testing methods"


class TestPapersToBibtex:
    def test_generates_combined(self) -> None:
        p1 = _make_paper(paper_id="s2-1", title="Paper A")
        p2 = _make_paper(paper_id="s2-2", title="Paper B", venue="ICML 2024")
        bib = papers_to_bibtex([p1, p2])
        assert bib.count("@") == 2
        assert "Paper A" in bib
        assert "Paper B" in bib


class TestSearchPapers:
    def test_search_papers_combines_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock both S2 and arXiv to verify combined search."""
        s2_paper = _make_paper(
            paper_id="s2-1", source="semantic_scholar", citation_count=100
        )
        arxiv_paper = _make_paper(
            paper_id="arxiv-1",
            title="Different Paper",
            doi="10.2/b",
            arxiv_id="2402.99999",
            source="arxiv",
            citation_count=50,
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_semantic_scholar",
            lambda *a, **kw: [s2_paper],
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_arxiv",
            lambda *a, **kw: [arxiv_paper],
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers("test", sources=["semantic_scholar", "arxiv"])
        assert len(papers) == 2
        # Should be sorted by citation_count desc
        assert papers[0].citation_count >= papers[1].citation_count

    def test_default_sources_arxiv_first(self) -> None:
        """arXiv should be the primary (first) source by default."""
        from researchclaw.literature.search import _DEFAULT_SOURCES
        assert _DEFAULT_SOURCES[0] == "arxiv"
        assert "semantic_scholar" in _DEFAULT_SOURCES

    def test_s2_failure_does_not_block_arxiv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When S2 fails, arXiv results should still be returned."""
        arxiv_paper = _make_paper(
            paper_id="arxiv-ok", title="ArXiv Paper", source="arxiv"
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_semantic_scholar",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("S2 down")),
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_arxiv",
            lambda *a, **kw: [arxiv_paper],
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers("test")
        assert len(papers) >= 1
        assert papers[0].source == "arxiv"

    def test_search_papers_unknown_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)
        papers = search_papers("test", sources=["unknown_source"])
        assert papers == []

    def test_search_papers_multi_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_search(*a: Any, **kw: Any) -> list[Paper]:
            nonlocal call_count
            call_count += 1
            return [
                _make_paper(
                    paper_id=f"s2-{call_count}",
                    title=f"Unique Paper {call_count}",
                    doi=f"10.{call_count}/unique",
                    arxiv_id=f"240{call_count}.{call_count:05d}",
                )
            ]

        monkeypatch.setattr(
            "researchclaw.literature.search.search_papers",
            mock_search,
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers_multi_query(["q1", "q2", "q3"])
        assert call_count == 3
        # All unique titles so no dedup
        assert len(papers) == 3


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_paper_with_no_meaningful_title_word(self) -> None:
        """cite_key should still work with stopword-only titles."""
        p = _make_paper(title="The And For With", year=2024)
        # All words are stopwords or <4 chars, keyword should be empty
        key = p.cite_key
        assert key.startswith("vaswani2024")

    def test_paper_multiple_authors_bibtex(self) -> None:
        p = _make_paper(
            authors=(
                Author(name="Alice One"),
                Author(name="Bob Two"),
                Author(name="Charlie Three"),
            )
        )
        bib = p.to_bibtex()
        assert "Alice One and Bob Two and Charlie Three" in bib

    def test_empty_s2_response(self) -> None:
        """_parse_s2_paper shouldn't crash on minimal data."""
        p = _parse_s2_paper({"paperId": "x"})
        assert p.paper_id == "s2-x"
        assert p.title == ""
        assert p.authors == ()
