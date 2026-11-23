"""Exploratory Lecture 05 diagnostic probes, separate from the locked holdout.

Commands:
  uv run python -m scripts.day5_development_probes --validate
  uv run python -m scripts.day5_development_probes --max-cases 2
  uv run python -m scripts.day5_development_probes

All methods share one query embedding per question. No answer-generation API.
Reference IDs are consulted only after retrieval, for metrics; never for selection.
The reference annotations are provisional and require manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data" / "day5_benchmark" / "development_probes.json"
DEFAULT_OUTPUT = ROOT / ".day5_benchmark_outputs" / "dev_probes_v1.jsonl"


def load_probes(path: Path = PROBES) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["probes"]


def validate_probes(probes: list[dict], known_page_ids: set[str]) -> None:
    if not probes:
        raise ValueError("No probes defined")
    seen: set[str] = set()
    for item in probes:
        pid = item.get("id")
        if not isinstance(pid, str) or not pid.startswith("P") or pid in seen:
            raise ValueError(f"Invalid/duplicate probe ID: {pid!r}")
        seen.add(pid)
        question = item.get("question")
        if not isinstance(question, str) or len(question.strip()) < 20:
            raise ValueError(f"Invalid question for {pid}")
        refs = item.get("reference_pages")
        if not isinstance(refs, list) or len(refs) < 2 or len(refs) != len(set(refs)):
            raise ValueError(f"Probe {pid} needs >=2 distinct reference pages")
        missing = set(refs) - known_page_ids
        if missing:
            raise ValueError(f"Unknown reference IDs for {pid}: {sorted(missing)}")


def vector_budget(vector_hits: list[dict], max_pages: int = 4, max_chars: int = 5000) -> list[dict]:
    """Strict vector-order baseline with a hard content budget and deduplication."""
    if max_pages <= 0 or max_chars <= 0:
        raise ValueError("Budgets must be positive")
    chosen: list[dict] = []
    seen: set[str] = set()
    used = 0
    for page in vector_hits:
        pid = page["page_id"]
        if pid in seen:
            continue
        seen.add(pid)
        size = len(str(page.get("content", "")))
        if not size or size + used > max_chars:
            continue
        chosen.append(page)
        used += size
        if len(chosen) >= max_pages:
            break
    return chosen


def summarise(pages: list[dict], gold: list[str]) -> dict[str, Any]:
    ids = list(dict.fromkeys(p["page_id"] for p in pages))
    hits = len(set(gold) & set(ids))
    return {
        "ids": ids,
        "count": len(ids),
        "chars": sum(len(str(p.get("content", ""))) for p in pages),
        "reference_hits": hits,
        "reference_total": len(gold),
        "reference_recall": hits / len(gold),
        "missing_reference_pages": sorted(set(gold) - set(ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Only validate probe annotations; no API calls")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.max_cases <= 6:
        parser.error("--max-cases must be from 1 to 6")

    # Intentionally never read the reserved question split.
    from app import retrieval
    from scripts.day5_benchmark import embedding_cache, load_corpus_pages
    from app.hybrid_retrieval import search_hybrid

    known = {p["page_id"] for p in load_corpus_pages()}
    probes = load_probes()
    validate_probes(probes, known)
    print(f"PASS: {len(probes)} exploratory probes verified against {len(known)} pages")
    if args.validate:
        return

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"Output already exists: {args.output}. Choose a fresh --output to avoid duplicate runs.")

    probe_sha = hashlib.sha256(PROBES.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(
        (ROOT / "app" / "retrieval.py").read_bytes()
        + (ROOT / "app" / "hybrid_retrieval.py").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()
    records: list[dict] = []
    with embedding_cache() as cache:
        for item in probes[: args.max_cases]:
            query = item["question"]
            start_calls = cache["calls"]
            embed_start = time.perf_counter()
            retrieval.embed_text(query.strip(), "RETRIEVAL_QUERY")
            embedding_seconds = time.perf_counter() - embed_start
            embedding_calls = cache["calls"] - start_calls

            def timed(fn):
                start = time.perf_counter()
                value = fn()
                return value, round(time.perf_counter() - start, 4)

            # Independently time each complete retrieval path with one shared
            # query embedding already warmed in the common cache.
            vector, tv = timed(lambda: retrieval.search(query))
            budgeted, tb = timed(lambda: vector_budget(retrieval.search(query)))
            graph, tg = timed(lambda: retrieval.search_graph(query))
            graph_pages = graph["seeds"] + graph["expanded"]
            fused, tf = timed(lambda: search_hybrid(query))
            hybrid_pages = fused["seeds"] + fused["expanded"]
            if cache["calls"] - start_calls != embedding_calls:
                raise RuntimeError("Method unexpectedly requested another embedding")
            gold = item["reference_pages"]
            mapped = {
                "vector": (vector, tv),
                "vector_budget": (budgeted, tb),
                "graph": (graph_pages, tg),
                "hybrid": (hybrid_pages, tf),
            }
            summary = {key: {**summarise(value, gold), "algorithm_seconds": secs}
                       for key, (value, secs) in mapped.items()}
            record = {
                "probe": item["id"],
                "category": item["category"],
                "question": query,
                "provisional_reference_pages": gold,
                "probe_set_sha256": probe_sha,
                "code_sha256": source_sha,
                "shared_query_embedding_seconds": round(embedding_seconds, 4),
                "shared_query_embedding_api_calls": embedding_calls,
                "methods": summary,
                "hybrid_graph_only_ids": sorted(set(summary["hybrid"]["ids"]) - set(summary["vector"]["ids"])),
                "hybrid_vs_vector_budget_same_ids": summary["hybrid"]["ids"] == summary["vector_budget"]["ids"],
                "graph_gold_not_in_vector_budget": sorted((set(gold) & set(summary["graph"]["ids"])) - set(summary["vector_budget"]["ids"])),
            }
            records.append(record)
            short = "  ".join(f"{k}={v['reference_hits']}/{v['reference_total']}" for k, v in summary.items())
            print(f"{item['id']}: {short}; graph-only hybrid={len(record['hybrid_graph_only_ids'])}; embedding_calls={embedding_calls}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Structured diagnostics: {args.output}")
    print("NOTE: Provisional reference-page recall is not answer correctness. No generation was performed.")


if __name__ == "__main__":
    main()
