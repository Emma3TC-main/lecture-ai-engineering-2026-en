"""Day 5 Advanced: budgeted, query-aware graph retrieval."""

import re

from app.retrieval import search_graph


STOP_WORDS = {
    "the", "and", "for", "from", "with", "what",
    "which", "when", "where", "this", "that",
    "does", "how", "are", "was", "were", "can",
    "all", "our", "into", "about", "their",
    "these", "those", "between", "lecture",
    "examples", "just", "built", "did", "you",
}


def terms(text: str) -> set[str]:
    """Extract distinct English query terms without external models."""
    return {
        word
        for word in re.findall(r"[a-z][a-z0-9]*", text.casefold())
        if len(word) >= 3 and word not in STOP_WORDS
    }


def select_graph_candidates(
    query: str,
    graph: dict,
    max_pages: int = 4,
    max_content_chars: int = 5000,
) -> dict:
    """Select nonduplicate graph pages within an explicit text budget."""
    if max_pages < 1 or max_content_chars < 1:
        raise ValueError("Budgets must be positive.")

    pool = []
    seen = set()

    for origin in ("seeds", "expanded"):
        for page in graph.get(origin, []):
            page_id = page["page_id"]

            if page_id in seen:
                continue

            seen.add(page_id)
            pool.append((origin, page))

    if not pool:
        return {
            "seeds": [],
            "expanded": [],
            "max_hops": graph.get("max_hops"),
            "selection": {"selected_pages": 0, "content_chars": 0},
        }

    wanted = terms(query)
    selected = []
    covered = set()
    used_chars = 0

    # Preserve ranked vector seeds before graph expansion.
    # Repeated query terms do not imply redundant evidence.
    remaining = []

    for origin, page in pool:
        size = len(page["content"])

        if (
            origin == "seeds"
            and len(selected) < max_pages
            and used_chars + size <= max_content_chars
        ):
            selected.append((origin, page))
            used_chars += size
            covered |= (
                terms(page["title"] + " " + page["content"])
                & wanted
            )
        else:
            remaining.append((origin, page))

    while len(selected) < max_pages:
        options = []

        for index, (origin, page) in enumerate(remaining):
            size = len(page["content"])

            if used_chars + size > max_content_chars:
                continue

            title_hits = terms(page["title"]) & wanted
            all_hits = (
                terms(page["title"] + " " + page["content"])
                & wanted
            )
            new_hits = all_hits - covered

            # Do not add pages with no overlap with the question.
            if not all_hits:
                continue

            # Prefer new coverage, title relevance and shorter paths.
            priority = (
                len(new_hits),
                len(title_hits),
                len(all_hits),
                -int(page.get("hops", 0)),
                int(origin == "seeds"),
                -index,
            )
            options.append((priority, index, origin, page, all_hits))

        if not options:
            break

        _, index, origin, page, hits = max(options)
        selected.append((origin, page))
        used_chars += len(page["content"])
        covered |= hits
        remaining.pop(index)

    seeds = [page for origin, page in selected if origin == "seeds"]
    expanded = [
        page for origin, page in selected if origin == "expanded"
    ]

    return {
        "seeds": seeds,
        "expanded": expanded,
        "max_hops": graph.get("max_hops"),
        "selection": {
            "strategy": "seed_first_graph_expansion_with_budget",
            "candidate_pages": len(pool),
            "selected_pages": len(selected),
            "content_chars": used_chars,
            "max_pages": max_pages,
            "max_content_chars": max_content_chars,
        },
    }


def search_graph_selective(query: str) -> dict:
    """Reuse original Graph candidates, then apply selective retrieval."""
    return select_graph_candidates(query, search_graph(query))
