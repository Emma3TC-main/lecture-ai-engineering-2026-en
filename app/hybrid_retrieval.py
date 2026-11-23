"""Lecture 05 v3a: vector/graph candidate fusion under a hard text budget.

This prototype intentionally does not read benchmark answers or reference pages.
It preserves the original search index and the official graph traversal.
"""

from __future__ import annotations

from app import retrieval


def fuse_candidates(
    vector_hits: list[dict],
    graph_result: dict,
    *,
    max_pages: int = 4,
    max_content_chars: int = 5000,
    rrf_k: int = 60,
) -> dict:
    """Fuse ranked vector/graph evidence with reciprocal rank fusion.

    Scores are calculated from candidate rankings, not gold/reference IDs.
    In equal-score ties, prefer vector rank, then graph rank, then page ID.
    """
    if max_pages <= 0 or max_content_chars <= 0 or rrf_k <= 0:
        raise ValueError("All budgets and rrf_k must be positive")

    pool: dict[str, dict] = {}

    def register(hits: list[dict], source: str) -> None:
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            page_id = hit["page_id"]
            if page_id in seen:
                continue
            seen.add(page_id)
            entry = pool.setdefault(
                page_id,
                {"page": dict(hit), "vector_rank": None, "graph_rank": None, "score": 0.0},
            )
            # The graph may provide path and hop metadata missing from vector results.
            if source == "graph":
                entry["page"] = {**entry["page"], **hit}
            entry[f"{source}_rank"] = rank
            entry["score"] += 1.0 / (rrf_k + rank)

    register(vector_hits, "vector")
    graph_hits = list(graph_result.get("seeds", [])) + list(graph_result.get("expanded", []))
    register(graph_hits, "graph")

    ordered = sorted(
        pool.values(),
        key=lambda entry: (
            -entry["score"],
            entry["vector_rank"] if entry["vector_rank"] is not None else float("inf"),
            entry["graph_rank"] if entry["graph_rank"] is not None else float("inf"),
            entry["page"]["page_id"],
        ),
    )

    chosen: list[dict] = []
    used_chars = 0
    for entry in ordered:
        page = entry["page"]
        size = len(str(page.get("content", "")))
        if size < 1 or used_chars + size > max_content_chars:
            continue
        chosen.append(entry)
        used_chars += size
        if len(chosen) >= max_pages:
            break

    # Graph instructions interpret "seeds" as vector starting points.
    seeds = [e["page"] for e in chosen if e["vector_rank"] is not None]
    expanded = [e["page"] for e in chosen if e["vector_rank"] is None]
    return {
        "seeds": seeds,
        "expanded": expanded,
        "max_hops": graph_result.get("max_hops"),
        "selection": {
            "strategy": "rrf_vector_graph_budget_v3a",
            "candidate_pages": len(pool),
            "selected_pages": len(chosen),
            "content_chars": used_chars,
            "max_pages": max_pages,
            "max_content_chars": max_content_chars,
            "vector_candidates": len(vector_hits),
            "graph_candidates": len(graph_hits),
        },
    }


def search_hybrid(query: str, *, max_pages: int = 4, max_content_chars: int = 5000) -> dict:
    """Embed once, then reuse ranked vector hits as graph seeds."""
    vector_hits = retrieval.search(query)
    graph = retrieval.search_graph(query, initial_hits=vector_hits)
    return fuse_candidates(
        vector_hits, graph, max_pages=max_pages, max_content_chars=max_content_chars
    )
