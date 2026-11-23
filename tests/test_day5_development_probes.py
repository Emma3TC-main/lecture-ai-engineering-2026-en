"""Offline unit tests for exploratory probes (no API calls)."""
import pytest
from scripts.day5_development_probes import vector_budget, summarise, validate_probes


def make_page(i, content="content"):
    return {"page_id": i, "content": content}


def test_budgeted_vector_deduplicates_and_respects_char_limit():
    hits = [make_page("a", "aaaa"), make_page("a", "aaaa"), make_page("b", "bbbbbb"), make_page("c", "c")]
    chosen = vector_budget(hits, max_pages=2, max_chars=5)
    assert [p["page_id"] for p in chosen] == ["a", "c"]
    assert summarise(chosen, ["a", "c"])["chars"] == 5


def test_budget_rejects_nonpositive_constraints():
    with pytest.raises(ValueError):
        vector_budget([], max_pages=0)
    with pytest.raises(ValueError):
        vector_budget([], max_chars=0)


def test_recall_uses_reference_ids_only_for_posthoc_metrics():
    r = summarise([make_page("a"), make_page("b")], ["a", "c"])
    assert r["reference_recall"] == .5
    assert r["missing_reference_pages"] == ["c"]


def test_validation_rejects_unknown_references():
    with pytest.raises(ValueError, match="Unknown reference"):
        validate_probes([{"id":"P01", "question":"This is a sufficiently long diagnostic question.", "reference_pages":["a", "z"]}], {"a"})


def test_validation_rejects_duplicate_ids():
    q = {"id":"P01", "question":"This is a sufficiently long diagnostic question.", "reference_pages":["a", "b"]}
    with pytest.raises(ValueError, match="Invalid/duplicate"):
        validate_probes([q, q], {"a", "b"})
