"""Day 5 Advanced selection tests. No API calls."""

from app.advanced_retrieval import select_graph_candidates


def page(page_id, title, content, hops=1):
    return {
        "page_id": page_id,
        "document_id": "lec02-main",
        "title": title,
        "content": content,
        "hops": hops,
        "citation": f"Lecture 2, p.{page_id}",
    }


def test_relevance_and_budget():
    graph = {
        "seeds": [
            page("p1", "LoRA", "LoRA freezes model weights", 0)
        ],
        "expanded": [
            page("p2", "QLoRA", "QLoRA uses four-bit weights"),
            page("p3", "Unrelated", "Completely unrelated topic"),
        ],
        "max_hops": 2,
    }

    result = select_graph_candidates(
        "How do LoRA and QLoRA differ?",
        graph,
        max_pages=2,
        max_content_chars=200,
    )

    ids = [
        p["page_id"]
        for p in result["seeds"] + result["expanded"]
    ]

    assert ids == ["p1", "p2"]
    assert result["selection"]["selected_pages"] == 2


def test_no_duplicates_and_preserve_citations():
    original = page("p1", "LoRA", "LoRA weights", 0)

    graph = {
        "seeds": [original],
        "expanded": [
            dict(original),
            page("p2", "QLoRA", "QLoRA quantization"),
        ],
        "max_hops": 2,
    }

    result = select_graph_candidates("LoRA QLoRA", graph)
    pages = result["seeds"] + result["expanded"]

    assert len({p["page_id"] for p in pages}) == len(pages)
    assert all("citation" in p for p in pages)


def test_budget_never_exceeded():
    graph = {
        "seeds": [page("p1", "LoRA", "LoRA weights", 0)],
        "expanded": [
            page("p2", "QLoRA", "QLoRA " * 300),
        ],
        "max_hops": 2,
    }

    result = select_graph_candidates(
        "LoRA QLoRA",
        graph,
        max_pages=3,
        max_content_chars=100,
    )

    assert result["selection"]["content_chars"] <= 100
    assert result["selection"]["selected_pages"] == 1


def test_preserves_primary_evidence_before_expansion():
    """Relevant vector seeds must not be replaced by weaker graph matches."""
    graph = {
        "seeds": [
            page("p65", "Context engineering",
                 "compaction memory notes sessions", 0),
            page("p38", "Introduction to memory",
                 "memory long-term context", 0),
            page("p39", "Compaction versus memory",
                 "compaction memory notes sessions", 0),
        ],
        "expanded": [
            page("p64", "Long tasks", "long tasks"),
            page("p40", "Daily notes",
                 "daily notes long-term memory"),
        ],
        "max_hops": 2,
    }

    result = select_graph_candidates(
        "compaction memory sessions daily notes long-term memory",
        graph,
        max_pages=4,
        max_content_chars=5000,
    )

    seed_ids = [p["page_id"] for p in result["seeds"]]
    expanded_ids = [p["page_id"] for p in result["expanded"]]

    assert seed_ids == ["p65", "p38", "p39"]
    assert expanded_ids == ["p40"]
    assert result["selection"]["selected_pages"] == 4
    assert result["selection"]["content_chars"] <= 5000
