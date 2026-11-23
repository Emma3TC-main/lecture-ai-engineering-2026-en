"""Day 5 adjacent retrieval tests. No Gemini API calls."""

from app import retrieval


def test_adjacent_pages_do_not_cross_documents(monkeypatch):
    pages = [
        {"page_id": "a1", "document_id": "A", "content": "A1"},
        {"page_id": "a2", "document_id": "A", "content": "A2"},
        {"page_id": "a3", "document_id": "A", "content": "A3"},
        {"page_id": "b1", "document_id": "B", "content": "B1"},
    ]

    monkeypatch.setattr(retrieval, "load_pages", lambda _: pages)
    monkeypatch.setattr(
        retrieval,
        "search",
        lambda _: [dict(pages[2])],
    )

    result = retrieval.search_with_neighbors("example")

    assert [p["page_id"] for p in result] == ["a3", "a2"]
    assert all(p["document_id"] == "A" for p in result)


def test_adjacent_pages_are_not_duplicated(monkeypatch):
    pages = [
        {"page_id": f"a{i}", "document_id": "A", "content": str(i)}
        for i in range(1, 5)
    ]

    monkeypatch.setattr(retrieval, "load_pages", lambda _: pages)
    monkeypatch.setattr(
        retrieval,
        "search",
        lambda _: [dict(pages[1]), dict(pages[2])],
    )

    result = retrieval.search_with_neighbors("example")
    ids = [p["page_id"] for p in result]

    assert ids == ["a2", "a3", "a1", "a4"]
    assert len(ids) == len(set(ids))
    assert result[0]["retrieval_origin"] == "vector"
    assert result[2]["retrieval_origin"] == "adjacent"
