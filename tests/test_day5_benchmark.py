"""Pure benchmark tests: no Gemini API, no local course import required."""

import copy

import pytest

from scripts.day5_benchmark import (
    DATASET_PATH, flatten_pages, load_dataset, mean_sd, parse_methods,
    retrieval_metrics, select_questions, validate_dataset,
)


def synthetic_corpus():
    dataset = load_dataset(DATASET_PATH)
    unique_ids = sorted({page for q in dataset["questions"] for page in q["reference_pages"]})
    return [{"page_id": identifier} for identifier in unique_ids]


def test_dataset_is_valid_against_its_references():
    summary = validate_dataset(load_dataset(), synthetic_corpus())
    assert summary["questions"] == 20
    assert summary["splits"] == {"development": 4, "holdout": 16}


def test_duplicate_ids_fail():
    dataset = load_dataset()
    bad = copy.deepcopy(dataset)
    bad["questions"][1]["id"] = bad["questions"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(bad, synthetic_corpus())


def test_unrecognized_reference_page_fails():
    dataset = load_dataset()
    bad = copy.deepcopy(dataset)
    bad["questions"][0]["reference_pages"].append("lec02-main-p9999")
    with pytest.raises(ValueError, match="Missing reference"):
        validate_dataset(bad, synthetic_corpus())


def test_split_and_limit_are_deterministic():
    dataset = load_dataset()
    assert [q["id"] for q in select_questions(dataset, "development", 2)] == ["D01", "D02"]
    assert len(select_questions(dataset, "holdout")) == 16
    assert len(select_questions(dataset, "all")) == 20


def test_vector_and_graph_normalization():
    p = {"page_id": "a", "content": "one"}
    assert flatten_pages([p]) == [p]
    assert flatten_pages({"seeds": [p], "expanded": []}) == [p]
    with pytest.raises(TypeError):
        flatten_pages("unexpected")


def test_retrieval_metrics_do_not_confuse_coverage_with_accuracy():
    got = retrieval_metrics(
        [{"page_id": "a", "content": "123"}, {"page_id": "b", "content": "45"}],
        ["a", "c"], True,
    )
    assert got["reference_recall"] == 0.5
    assert got["content_chars"] == 5
    assert got["all_reference_pages_found"] is False


def test_unanswerable_does_not_get_perfect_recall():
    got = retrieval_metrics([{"page_id": "a", "content": "wrong"}], [], False)
    assert got["reference_recall"] is None
    assert got["all_reference_pages_found"] is None


def test_invalid_duplicate_methods_rejected():
    with pytest.raises(ValueError):
        parse_methods("vector,vector")
    with pytest.raises(ValueError):
        parse_methods("magic")
    assert parse_methods("vector,selective") == ["vector", "selective"]


def test_variance_not_fabricated_for_one_observation():
    assert "no variance estimate" in mean_sd([2.5])
    assert "n=2" in mean_sd([2.0, 4.0])
    assert mean_sd([]) == "N/A"


def test_holdout_cannot_run_accidentally():
    from argparse import Namespace
    from scripts.day5_benchmark import guard_holdout

    with pytest.raises(SystemExit, match="Holdout locked"):
        guard_holdout(Namespace(split="holdout", unlock_holdout=False))
    guard_holdout(Namespace(split="holdout", unlock_holdout=True))


def test_live_requires_explicit_api_opt_in():
    from argparse import Namespace
    from scripts.day5_benchmark import run_live

    with pytest.raises(SystemExit, match="--allow-api"):
        run_live(Namespace(allow_api=False))


def test_jsonl_roundtrip(tmp_path):
    from scripts.day5_benchmark import append_jsonl, iter_jsonl

    path = tmp_path / "output.jsonl"
    append_jsonl(path, {"id": "D01", "error": "prueba", "status": "ok"})
    append_jsonl(path, {"id": "D02", "status": "error"})
    assert [record["id"] for record in iter_jsonl(path)] == ["D01", "D02"]


def test_retrieval_methods_share_one_query_embedding_outside_timers(monkeypatch, tmp_path):
    """Fair timing: a query is embedded once, then each method runs warm."""
    from argparse import Namespace
    from scripts import day5_benchmark as bm
    from app import retrieval

    queries = []

    def fake_embed(text, task):
        queries.append((text, task))
        return [1.0, 0.0]

    monkeypatch.setattr(retrieval, "embed_text", fake_embed)
    monkeypatch.setattr(bm, "load_corpus_pages", lambda: [])
    monkeypatch.setattr(bm, "validate_dataset", lambda *_: {})
    monkeypatch.setattr(bm, "git_state", lambda: {"commit": "test", "dirty": True})
    monkeypatch.setattr(bm, "source_fingerprint", lambda: {"code_sha256": "test"})

    def fake_retrieve(method, question):
        retrieval.embed_text(question.strip(), "RETRIEVAL_QUERY")
        return [{"page_id": "lec02-main-p051", "content": "sample"},
                {"page_id": "lec02-main-p052", "content": "sample"}]

    monkeypatch.setattr(bm, "retrieve_method", fake_retrieve)
    output = tmp_path / "retrieval.jsonl"
    args = Namespace(split="development", unlock_holdout=False,
                     max_cases=1, methods="vector,neighbors,graph,selective",
                     output=str(output), force=False)
    bm.run_retrieve(args)

    records = list(bm.iter_jsonl(output))
    assert len(records) == 4
    assert len(queries) == 1
    assert queries[0][1] == "RETRIEVAL_QUERY"
    assert all(r["embedding_api_calls"] == 0 for r in records)
    assert all(r["shared_query_embedding_api_calls"] == 1 for r in records)
    assert all(r["timing_mode"] == "warm_query_embedding_cached" for r in records)
