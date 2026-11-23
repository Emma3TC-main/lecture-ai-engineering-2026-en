"""Tests for the v3a retrieval candidate fusion. No model/API calls."""

import pytest

from app.hybrid_retrieval import fuse_candidates, search_hybrid


def page(id, chars=500, *, citation=None):
    return {
        "page_id": id,
        "document_id": id.split("-p")[0],
        "title": id,
        "content": "a" * chars,
        "citation": citation or id,
    }


def ids(result):
    return [p["page_id"] for p in result["seeds"] + result["expanded"]]


def test_d04_retains_fourth_vector_hit_missing_from_graph():
    vector = [
        page("lec02-main-p052", 1238),
        page("lec02-main-p051", 1349),
        page("lec03-main-p065", 1076),
        page("lec03-main-p039", 846),
        page("lec02-main-p060", 875),
    ]
    graph = {
        "seeds": vector[:3],
        "expanded": [
            page("lec02-main-p053"),
            page("lec02-main-p050"),
            page("lec03-main-p064"),
            page("lec03-main-p066"),
        ],
        "max_hops": 2,
    }
    result = fuse_candidates(vector, graph)
    assert set(ids(result)) == {
        "lec02-main-p052", "lec02-main-p051",
        "lec03-main-p065", "lec03-main-p039",
    }
    assert result["selection"]["content_chars"] == 4509
    assert result["selection"]["selected_pages"] == 4


def test_graph_expansion_selected_when_vector_candidates_too_large():
    vector = [page("d-p1", 300), page("d-p2", 3000)]
    graph = {"seeds": vector[:1], "expanded": [page("d-p3", 200)]}
    result = fuse_candidates(vector, graph, max_pages=2, max_content_chars=600)
    assert ids(result) == ["d-p1", "d-p3"]
    assert result["expanded"][0]["citation"] == "d-p3"


def test_deduplicates_and_preserves_metadata():
    vector = [page("d-p1", 100, citation="Lecture 3 p1")]
    graph = {"seeds": [page("d-p1", 100, citation="Lecture 3 p1")], "expanded": [page("d-p1", 100)]}
    result = fuse_candidates(vector, graph)
    assert ids(result) == ["d-p1"]
    assert result["seeds"][0]["citation"] == "Lecture 3 p1"
    assert result["selection"]["candidate_pages"] == 1


@pytest.mark.parametrize("kwargs", [
    {"max_pages": 0}, {"max_content_chars": 0}, {"rrf_k": 0}
])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        fuse_candidates([], {}, **kwargs)


def test_single_embedding_path_uses_initial_hits(monkeypatch):
    vector = [page("d-p1")]
    calls = {"search": 0, "graph": 0}

    def fake_search(query):
        calls["search"] += 1
        return vector

    def fake_graph(query, *, initial_hits):
        calls["graph"] += 1
        assert initial_hits is vector
        return {"seeds": vector, "expanded": []}

    monkeypatch.setattr("app.hybrid_retrieval.retrieval.search", fake_search)
    monkeypatch.setattr("app.hybrid_retrieval.retrieval.search_graph", fake_graph)
    result = search_hybrid("question")
    assert ids(result) == ["d-p1"]
    assert calls == {"search": 1, "graph": 1}
