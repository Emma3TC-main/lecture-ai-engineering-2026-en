"""AIE2026 Lecture 05: reproducible retrieval and answer evaluation.

No model or embedding API is used by `validate`, `report`, or `review-template`.
`retrieve` calls the Gemini embedding endpoint; `live` calls the embedding and
answer-generation endpoints. Never supply reference answers to an agent.

Examples:
    uv run python -m scripts.day5_benchmark validate
    uv run python -m scripts.day5_benchmark retrieve --split development --max-cases 2
    uv run python -m scripts.day5_benchmark live --allow-api --split development --max-cases 1 --methods vector,selective
    uv run python -m scripts.day5_benchmark report
    uv run python -m scripts.day5_benchmark review-template
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "day5_benchmark" / "questions.json"
OUTPUT_DIR = ROOT / ".day5_benchmark_outputs"
METHODS = ("vector", "neighbors", "graph", "selective")
SPLITS = ("development", "holdout", "all")
CATEGORIES = {"single_topic", "multi_page", "cross_lecture", "unanswerable", "paraphrase"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain"))}


def load_dataset(path: Path = DATASET_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(dataset: dict, pages: list[dict]) -> dict:
    """Validate a fixed benchmark and its reference page IDs against the real source."""
    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty array")

    page_ids = [page["page_id"] for page in pages]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Duplicate page IDs in corpus")
    valid_pages = set(page_ids)
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for q in questions:
        q_id = q.get("id")
        if not isinstance(q_id, str) or not q_id or q_id in seen:
            raise ValueError(f"Duplicate or missing question ID: {q_id}")
        seen.add(q_id)
        split = q.get("split")
        category = q.get("category")
        if split not in ("development", "holdout") or category not in CATEGORIES:
            raise ValueError(f"Invalid split/category for {q_id}")
        if not isinstance(q.get("question"), str) or len(q["question"].strip()) < 15:
            raise ValueError(f"Question missing or too short: {q_id}")
        gold = q.get("reference_pages")
        if not isinstance(gold, list) or len(gold) != len(set(gold)):
            raise ValueError(f"Reference pages must be a unique list: {q_id}")
        missing = set(gold) - valid_pages
        if missing:
            raise ValueError(f"Missing reference pages for {q_id}: {sorted(missing)}")
        if not isinstance(q.get("answerable"), bool):
            raise ValueError(f"Invalid answerable field: {q_id}")
        if q["answerable"] != bool(gold):
            raise ValueError(f"Answerability/reference page mismatch: {q_id}")
        if (category == "unanswerable") != (not q["answerable"]):
            raise ValueError(f"Unanswerable category mismatch: {q_id}")
        if not isinstance(q.get("expected_claims"), list) or not q["expected_claims"]:
            raise ValueError(f"Expected claims missing: {q_id}")
        split_counts[split] += 1
        category_counts[category] += 1

    return {
        "questions": len(questions),
        "pages": len(pages),
        "splits": dict(split_counts),
        "categories": dict(category_counts),
    }


def select_questions(dataset: dict, split: str, max_cases: int | None = None) -> list[dict]:
    questions = [q for q in dataset["questions"] if split == "all" or q["split"] == split]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("--max-cases must be >= 1")
        questions = questions[:max_cases]
    return questions


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",")]
    if not methods or any(part not in METHODS for part in methods) or len(methods) != len(set(methods)):
        raise ValueError(f"Methods must be unique names from: {', '.join(METHODS)}")
    return methods


def flatten_pages(result: Any) -> list[dict]:
    """Normalize original return values without silently dropping duplicated IDs."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return list(result.get("seeds", [])) + list(result.get("expanded", []))
    raise TypeError(f"Unsupported retrieval result: {type(result).__name__}")


def retrieval_metrics(pages: list[dict], reference_pages: list[str], answerable: bool) -> dict:
    ids = list(dict.fromkeys(p["page_id"] for p in pages))
    gold = set(reference_pages)
    overlap = gold.intersection(ids)
    return {
        "retrieved_page_ids": ids,
        "retrieved_pages": len(ids),
        "content_chars": sum(len(str(p.get("content", ""))) for p in pages),
        "reference_page_hits": len(overlap),
        "reference_page_total": len(gold),
        "reference_recall": len(overlap) / len(gold) if answerable and gold else None,
        "all_reference_pages_found": gold.issubset(ids) if answerable else None,
    }


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {e}") from e


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def load_corpus_pages() -> list[dict]:
    """Import course modules only when reading the local corpus."""
    from app.retrieval import load_pages

    return load_pages()


def source_fingerprint() -> dict:
    from app.config import RAG_DB, RAG_SOURCE

    # Fingerprint code as well as data: uncommitted algorithm revisions must
    # never be mistaken for the same experiment merely because HEAD is unchanged.
    paths = [
        ROOT / "app" / "retrieval.py",
        ROOT / "app" / "advanced_retrieval.py",
        ROOT / "app" / "agent.py",
        ROOT / "app" / "config.py",
        ROOT / "scripts" / "day5_benchmark.py",
    ]
    code_hasher = hashlib.sha256()
    for path in paths:
        code_hasher.update(str(path.relative_to(ROOT)).encode("utf-8"))
        code_hasher.update(path.read_bytes() if path.is_file() else b"MISSING")
    return {
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "materials_sha256": hashlib.sha256(RAG_SOURCE.read_bytes()).hexdigest(),
        "rag_db_sha256": hashlib.sha256(RAG_DB.read_bytes()).hexdigest(),
        "code_sha256": code_hasher.hexdigest(),
    }


def retrieve_method(method: str, query: str) -> list[dict]:
    from app import retrieval

    if method == "vector":
        return retrieval.search(query)
    if method == "neighbors":
        return retrieval.search_with_neighbors(query, radius=1)
    if method == "graph":
        return flatten_pages(retrieval.search_graph(query))
    if method == "selective":
        from app.advanced_retrieval import search_graph_selective

        return flatten_pages(search_graph_selective(query))
    raise ValueError(f"Unknown retrieval method: {method}")


@contextmanager
def embedding_cache() -> Iterator[dict]:
    """Deduplicate identical query embeddings in the SAME retrieval-only process.

    This does not cache, save, or modify the 214-page index. A new process
    creates a new cache and may call the embedding API again.
    """
    from app import retrieval

    original = retrieval.embed_text
    misses = {"calls": 0}

    @lru_cache(maxsize=256)
    def cached(text: str, task: str):
        misses["calls"] += 1
        return original(text, task)

    retrieval.embed_text = cached
    try:
        yield misses
    finally:
        retrieval.embed_text = original


def run_validate(_: argparse.Namespace) -> None:
    summary = validate_dataset(load_dataset(), load_corpus_pages())
    print("PASS: Benchmark schema, IDs, and corpus references validated (no API calls).")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def guard_holdout(args: argparse.Namespace) -> None:
    if args.split in ("holdout", "all") and not args.unlock_holdout:
        raise SystemExit("Holdout locked: commit/freeze your algorithm first, then add --unlock-holdout.")


def run_retrieve(args: argparse.Namespace) -> None:
    guard_holdout(args)
    dataset = load_dataset()
    validate_dataset(dataset, load_corpus_pages())
    questions = select_questions(dataset, args.split, args.max_cases)
    methods = parse_methods(args.methods)
    output = Path(args.output)
    state = git_state()
    fingerprints = source_fingerprint()
    completed = {(r.get("id"), r.get("method")) for r in iter_jsonl(output) if r.get("status") == "ok"
                 and r.get("dataset_sha256") == fingerprints["dataset_sha256"]
                 and r.get("materials_sha256") == fingerprints["materials_sha256"]
                 and r.get("rag_db_sha256") == fingerprints["rag_db_sha256"]
                 and r.get("code_sha256") == fingerprints["code_sha256"]}
    print(f"Retrieval-only: {len(questions)} question(s), {len(methods)} method(s); embedding API may be called.")
    # Warm the query embedding ONCE outside every retrieval timer. This
    # makes algorithm timings comparable instead of charging the first
    # method with Gemini API latency and treating subsequent methods as free.
    from app import retrieval
    with embedding_cache() as cache:
        for question in questions:
            pending = [method for method in methods
                       if args.force or (question["id"], method) not in completed]
            if not pending:
                print(f"SKIP {question['id']}: all requested methods recorded")
                continue
            embedding_start = time.perf_counter()
            initial_embedding_calls = cache["calls"]
            retrieval.embed_text(question["question"].strip(), "RETRIEVAL_QUERY")
            shared_embedding_seconds = round(time.perf_counter() - embedding_start, 4)
            shared_embedding_calls = cache["calls"] - initial_embedding_calls
            for method in methods:
                if not args.force and (question["id"], method) in completed:
                    print(f"SKIP {question['id']} / {method}: successful record already exists")
                    continue
                initial_calls = cache["calls"]
                started = time.perf_counter()
                record = {
                    "timestamp_utc": utc_now(), "phase": "retrieval", "id": question["id"],
                    "split": question["split"], "category": question["category"],
                    "method": method, "answerable": question["answerable"],
                    "reference_pages": question["reference_pages"],
                    "model": None, "git_commit": state["commit"], "git_dirty": state["dirty"],
                    "timing_mode": "warm_query_embedding_cached",
                    "shared_query_embedding_seconds": shared_embedding_seconds,
                    "shared_query_embedding_api_calls": shared_embedding_calls,
                    **fingerprints,
                }
                try:
                    pages = retrieve_method(method, question["question"])
                    record.update(retrieval_metrics(pages, question["reference_pages"], question["answerable"]))
                    record["status"] = "ok"
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
                record["embedding_api_calls"] = cache["calls"] - initial_calls
                # This is only an estimate for an uncached query: it sums
                # separate timed stages, not a measured end-to-end run.
                record["estimated_cold_total_seconds"] = round(
                    record["elapsed_seconds"] + shared_embedding_seconds, 4
                )
                append_jsonl(output, record)
                print(f"{record['status'].upper()} {question['id']:3s} {method:9s} {record['elapsed_seconds']:.2f}s "
                      f"pages={record.get('retrieved_pages', '-')} "
                      f"ref_recall={record.get('reference_recall')} "
                      f"embedding_calls={record['embedding_api_calls']}")
                if record["status"] == "error":
                    print(f"STOP: {record['error']} (no automatic API retry)", file=sys.stderr)
                    raise SystemExit(1)
    print(f"Structured results: {output}")


def tool_page_ids(result) -> tuple[list[str], list[dict]]:
    """Read tool metadata from agent trace, without retaining raw retrieved texts."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    ids: list[str] = []
    calls: list[dict] = []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls.append({"name": part.tool_name, "arguments": part.args})
            elif isinstance(part, ToolReturnPart):
                try:
                    data = json.loads(part.content) if isinstance(part.content, str) else part.content
                    pages = flatten_pages(data["results"] if isinstance(data, dict) and "results" in data else data)
                    ids.extend(p["page_id"] for p in pages if isinstance(p, dict) and "page_id" in p)
                except (KeyError, ValueError, TypeError):
                    pass
    return list(dict.fromkeys(ids)), calls


def build_agent(method: str):
    """Keep the official instructions and tool names for a fair comparison."""
    from pydantic_ai import Agent
    from app.agent import INSTRUCTIONS, GRAPH_INSTRUCTIONS

    if method in ("vector", "neighbors"):
        instructions = INSTRUCTIONS

        def search_materials(query: str) -> str:
            """Search the main materials for Lectures 01–03 and return relevant page text, titles, and page numbers.

            Use query to specify what you want to look up.
            """
            return json.dumps({"results": retrieve_method(method, query)}, ensure_ascii=False)

        agent = Agent(instructions=instructions)
        agent.tool_plain(search_materials)
        return agent

    if method in ("graph", "selective"):
        instructions = GRAPH_INSTRUCTIONS

        def search_graph(query: str) -> str:
            """Search the materials, then follow adjacent pages and shared terms to return related page text and citation metadata.

            Use query to specify what you want to look up.
            """
            if method == "graph":
                from app.retrieval import search_graph as original_graph
                return json.dumps(original_graph(query), ensure_ascii=False)
            from app.advanced_retrieval import search_graph_selective
            return json.dumps(search_graph_selective(query), ensure_ascii=False)

        agent = Agent(instructions=instructions)
        agent.tool_plain(search_graph)
        return agent
    raise ValueError(method)


def run_live(args: argparse.Namespace) -> None:
    if not args.allow_api:
        raise SystemExit("Live generation is disabled by default. Add --allow-api explicitly.")
    guard_holdout(args)
    if args.repeats < 1 or args.repeats > 5:
        raise SystemExit("--repeats must be in [1, 5]")
    if args.cooldown_seconds < 0:
        raise SystemExit("--cooldown-seconds must be >= 0")
    if args.cooldown_seconds < 30:
        print("WARNING: Short pauses may trigger Gemini free-tier rate limits.")
    from app.agent import build_model
    from app.config import MODEL
    from pydantic_ai.exceptions import ModelHTTPError

    dataset = load_dataset()
    validate_dataset(dataset, load_corpus_pages())
    questions = select_questions(dataset, args.split, args.max_cases)
    methods = parse_methods(args.methods)
    output = Path(args.output)
    fingerprints = source_fingerprint()
    state = git_state()
    already = {(r.get("id"), r.get("method"), r.get("repeat")) for r in iter_jsonl(output)
               if r.get("status") == "ok" and r.get("model") == MODEL
               and r.get("dataset_sha256") == fingerprints["dataset_sha256"]
               and r.get("materials_sha256") == fingerprints["materials_sha256"]
               and r.get("rag_db_sha256") == fingerprints["rag_db_sha256"]
               and r.get("code_sha256") == fingerprints["code_sha256"]}

    jobs = [(q, method, repeat) for q in questions for method in methods for repeat in range(1, args.repeats + 1)
            if (q["id"], method, repeat) not in already]
    print(f"LIVE API: {len(jobs)} job(s); model={MODEL}; cooldown={args.cooldown_seconds}s; no automatic retries.")
    if args.max_cases > 2:
        print("CAUTION: More than two cases selected; check your API quota before proceeding.")

    for index, (question, method, repeat) in enumerate(jobs):
        started = time.perf_counter()
        record = {
            "timestamp_utc": utc_now(), "phase": "generation", "id": question["id"],
            "split": question["split"], "category": question["category"],
            "method": method, "repeat": repeat, "answerable": question["answerable"],
            "reference_pages": question["reference_pages"],
            "model": MODEL, "git_commit": state["commit"], "git_dirty": state["dirty"],
            **fingerprints,
        }
        try:
            agent = build_agent(method)
            result = agent.run_sync(question["question"], model=build_model())
            record["status"] = "ok"
            record["answer"] = str(result.output)
            record["input_tokens"] = result.usage.input_tokens
            record["output_tokens"] = result.usage.output_tokens
            record["model_responses"] = result.usage.requests
            record["generation_cost_usd"] = float(result.usage.cost) if result.usage.cost is not None else None
            ids, calls = tool_page_ids(result)
            record["tool_calls"] = calls
            record["retrieved_page_ids"] = ids
            record["reference_recall"] = retrieval_metrics(
                [{"page_id": pid} for pid in ids], question["reference_pages"], question["answerable"]
            )["reference_recall"]
        except ModelHTTPError as exc:
            record.update(status="error", error=f"HTTP {exc.status_code}: {str(exc.body)[:500]}")
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        append_jsonl(output, record)
        print(f"{record['status'].upper()} {question['id']}/{method}/run{repeat} "
              f"time={record['elapsed_seconds']:.2f}s "
              f"tokens={record.get('input_tokens', '-')}/{record.get('output_tokens', '-')} "
              f"cost={record.get('generation_cost_usd')}")
        if record["status"] != "ok":
            print(f"STOP: {record['error']} (no automatic retry)", file=sys.stderr)
            raise SystemExit(1)
        if index < len(jobs) - 1 and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
    print(f"Structured results: {output}")


def mean_sd(values: list[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1; no variance estimate)"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f} (n={len(values)})"


def run_report(args: argparse.Namespace) -> None:
    for label, path in (("retrieval", Path(args.retrieval)), ("generation", Path(args.generation))):
        records = list(iter_jsonl(path))
        if not records:
            print(f"{label}: no records at {path}")
            continue
        print(f"\n=== {label.upper()} ({path}) ===")
        successful = [r for r in records if r.get("status") == "ok"]
        failures = [r for r in records if r.get("status") != "ok"]
        print(f"Records: {len(records)} | successful: {len(successful)} | failures: {len(failures)}")
        if label == "retrieval":
            modes = {r.get("timing_mode", "original_cache_order") for r in successful}
            if len(modes) > 1:
                print("WARNING: Mixed retrieval timing protocols; do not aggregate these records.")
            if "warm_query_embedding_cached" in modes:
                print("TIMING: Retrieval latency is warm-cache algorithm time; query embedding is measured separately.")
        by_method: dict[str, list] = defaultdict(list)
        for r in successful:
            by_method[r["method"]].append(r)
        for method, items in sorted(by_method.items()):
            times = [float(r["elapsed_seconds"]) for r in items if r.get("elapsed_seconds") is not None]
            costs = [float(r["generation_cost_usd"]) for r in items if r.get("generation_cost_usd") is not None]
            token_counts = [float(r["input_tokens"]) for r in items if r.get("input_tokens") is not None]
            recalls = [float(r["reference_recall"]) for r in items if r.get("reference_recall") is not None]
            counts = [float(r["retrieved_pages"]) for r in items if "retrieved_pages" in r]
            chars = [float(r["content_chars"]) for r in items if "content_chars" in r]
            print(f"{method:9s}: latency {mean_sd(times)}; retrieved pages {mean_sd(counts)}; "
                  f"content chars {mean_sd(chars)}; input tokens {mean_sd(token_counts)}; "
                  f"generation cost USD {mean_sd(costs)}; ref-page recall {mean_sd(recalls)}")
        print("NOTE: Reference-page recall is retrieval coverage, NOT answer accuracy or citation correctness.")
        if failures:
            print("Failures (excluded from success aggregates):")
            for r in failures:
                print(f"  {r.get('id')} {r.get('method')} {r.get('error', '')[:120]}")


def run_review_template(args: argparse.Namespace) -> None:
    records = [r for r in iter_jsonl(Path(args.generation)) if r.get("status") == "ok"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise SystemExit(f"Review sheet already exists: {output}. Pass --force to overwrite deliberately.")
    cols = ["id", "split", "method", "repeat", "answerable", "correctness_0_2", "grounding_0_2",
            "citations_0_2", "abstention_0_2_or_blank", "reviewer", "notes"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in records:
            writer.writerow({key: r[key] for key in ("id", "split", "method", "repeat", "answerable")})
    print(f"Created {len(records)} unscored human-review rows at {output}.")
    print("0=incorrect/unsupported, 1=partial, 2=fully correct; leave absent-case nonapplicable ratings blank.")
    print("Do NOT treat blank labels as zeros or report an accuracy score before review.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Schema and reference page check; zero API calls")

    retrieval = commands.add_parser("retrieve", help="Retrieval benchmark; calls Gemini embeddings")
    retrieval.add_argument("--split", choices=SPLITS, default="development")
    retrieval.add_argument("--unlock-holdout", action="store_true")
    retrieval.add_argument("--max-cases", type=int, default=2)
    retrieval.add_argument("--methods", default=",".join(METHODS))
    retrieval.add_argument("--force", action="store_true", help="Rerun even if a successful record exists")
    retrieval.add_argument("--output", default=str(OUTPUT_DIR / "retrieval.jsonl"))

    live = commands.add_parser("live", help="Generation benchmark; EXPLICIT API opt-in required")
    live.add_argument("--allow-api", action="store_true")
    live.add_argument("--split", choices=SPLITS, default="development")
    live.add_argument("--unlock-holdout", action="store_true")
    live.add_argument("--max-cases", type=int, default=1)
    live.add_argument("--methods", default="vector,selective")
    live.add_argument("--repeats", type=int, default=1)
    live.add_argument("--cooldown-seconds", type=float, default=40)
    live.add_argument("--output", default=str(OUTPUT_DIR / "generation.jsonl"))

    report = commands.add_parser("report", help="Summary; no API, no automatic correctness scoring")
    report.add_argument("--retrieval", default=str(OUTPUT_DIR / "retrieval.jsonl"))
    report.add_argument("--generation", default=str(OUTPUT_DIR / "generation.jsonl"))

    review = commands.add_parser("review-template", help="Blank human grading spreadsheet from live runs")
    review.add_argument("--generation", default=str(OUTPUT_DIR / "generation.jsonl"))
    review.add_argument("--output", default=str(OUTPUT_DIR / "human_review.csv"))
    review.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "validate": run_validate,
        "retrieve": run_retrieve,
        "live": run_live,
        "report": run_report,
        "review-template": run_review_template,
    }
    try:
        commands[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
