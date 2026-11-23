"""Deterministic unit tests of the offline candidate diagnostic."""
import pytest

from scripts.day5_candidate_probe import BM25, coverage, diagnose, interleave_unique, split_aspects


def page(id, title, content):
    return {"page_id": id, "title": title, "content": content}


def test_bm25_returns_relevant_results():
    bm25 = BM25([page("a", "Memory", "persistent session notes"),
                 page("b", "Quantization", "4 bit weights"),
                 page("c", "Memory details", "session compaction history")])
    assert bm25.rank("session memory", top_k=1)[0]["page_id"] == "a"


def test_bm25_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        BM25([])
    with pytest.raises(ValueError):
        BM25([page("a", "X", "Y")], title_boost=0)
    with pytest.raises(ValueError):
        BM25([page("a", "X", "Y")]).rank("X", top_k=0)


def test_subquery_split_is_deterministic():
    q = "How is long-task capability different from error accumulation in agents?"
    assert split_aspects(q) == ["How is long-task capability", "error accumulation in agents"]
    assert split_aspects("What is memory?") == ["What is memory?"]


def test_interleaving_preserves_per_query_rank_without_duplicates():
    rankings = [
        [{"page_id": "a"}, {"page_id": "c"}],
        [{"page_id": "b"}, {"page_id": "a"}],
    ]
    assert interleave_unique(rankings) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        interleave_unique(rankings, limit=0)


def test_gold_reference_cannot_affect_ranking():
    ps = [page("a", "Memory", "persistent notes"), page("b", "LoRA", "trained adapter")]
    q1 = {"id": "P01", "question": "How does memory persist between sessions?", "reference_pages": ["a"]}
    q2 = {**q1, "reference_pages": ["b"]}
    one, = diagnose(ps, [q1])
    two, = diagnose(ps, [q2])
    assert one["bm25_full"]["ids"] == two["bm25_full"]["ids"]
    assert one["bm25_aspects"]["ids"] == two["bm25_aspects"]["ids"]
    assert one["bm25_full"]["hits"] != two["bm25_full"]["hits"]


def test_coverage_is_posthoc_and_no_duplication():
    assert coverage(["a", "b"], ["a", "z"]) == {"hits": 1, "total": 2, "missing": ["z"]}
