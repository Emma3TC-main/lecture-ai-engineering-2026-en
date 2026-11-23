"""Lecture 05 development-only, API-free candidate generation diagnostics.

Compares BM25 full-query and naive aspect-wise BM25. This is an exploratory
candidate-coverage experiment, *not* a final RAG reranker or an answer judge.
No reference IDs are read by the retrieval/scoring routines. Holdout never loaded.

Example:
    uv run python -m scripts.day5_candidate_probe \
        --input .day5_benchmark_outputs/dev_probes_full.jsonl \
        --output .day5_benchmark_outputs/candidate_probe_v1.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "data/day5_benchmark/development_probes.json"
DEFAULT_INPUT = ROOT / ".day5_benchmark_outputs/dev_probes_full.jsonl"
DEFAULT_OUTPUT = ROOT / ".day5_benchmark_outputs/candidate_probe_v1.jsonl"

STOP = frozenset("""
a an and are as at be been but by can could did do does each for from had has have
how if in into is it its of on or our should so than that the their them there these
this those to was were what when where which who why will with would you your
""".split())


def tokens(value: str) -> list[str]:
    """Lightweight English tokenization with accent normalization, no package deps."""
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z][a-z0-9]*", text)
            if len(t) > 2 and t not in STOP]


def split_aspects(question: str) -> list[str]:
    """Blind string-based splitting; does not access ground-truth pages.

    Keep the original question separately during evaluation, because splits are
    syntactic rather than semantic and can sometimes make queries worse.
    """
    parts = re.split(
        r"\b(?:and also|and what|and when|different from|compared with|compared to|versus)\b|\bvs\.?\b",
        question.strip().rstrip("?"),
        flags=re.IGNORECASE,
    )
    parts = [p.strip(" ,;:.?\n\t") for p in parts if len(tokens(p)) >= 2]
    return parts if len(parts) >= 2 else [question.strip()]


class BM25:
    """Lexical retrieval over the local 214-page corpus; never uses API calls."""

    def __init__(self, pages: list[dict], *, title_boost: int = 2):
        if title_boost < 1:
            raise ValueError("title_boost must be >= 1")
        if not pages:
            raise ValueError("Corpus cannot be empty")
        self.pages = pages
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()
        for p in pages:
            words = tokens(str(p.get("title", ""))) * title_boost + tokens(str(p.get("content", "")))
            counts = Counter(words)
            self.docs.append(counts)
            self.lengths.append(sum(counts.values()))
            self.df.update(counts.keys())
        self.avgdl = sum(self.lengths) / len(pages)

    def rank(self, query: str, *, top_k: int = 10) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        q_terms = set(tokens(query))
        if not q_terms:
            return []
        n_docs = len(self.pages)
        results = []
        for p, tf, doc_len in zip(self.pages, self.docs, self.lengths):
            score = 0.0
            for term in q_terms:
                count = tf.get(term, 0)
                if not count:
                    continue
                freq = self.df[term]
                idf = math.log(1.0 + (n_docs - freq + .5) / (freq + .5))
                score += idf * (count * 2.2) / (count + 1.2 * (.25 + .75 * doc_len / self.avgdl))
            if score > 0:
                results.append({"page_id": p["page_id"], "score": round(score, 8)})
        results.sort(key=lambda x: (-x["score"], x["page_id"]))
        return results[:top_k]


def interleave_unique(rankings: list[list[dict]], *, limit: int = 12) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be positive")
    seen: set[str] = set()
    out: list[str] = []
    for rank in range(max((len(x) for x in rankings), default=0)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            page_id = ranking[rank]["page_id"]
            if page_id not in seen:
                seen.add(page_id)
                out.append(page_id)
                if len(out) >= limit:
                    return out
    return out


def coverage(ids: list[str], refs: list[str]) -> dict:
    # For reporting only; cannot influence ranking, decomposition or selection.
    found = set(ids) & set(refs)
    return {"hits": len(found), "total": len(refs), "missing": sorted(set(refs) - set(ids))}


def diagnose(pages: list[dict], probes: list[dict], *, top_k: int = 10, limit: int = 12) -> list[dict]:
    bm25 = BM25(pages)
    results = []
    for probe in probes:
        q = probe["question"]
        aspects = split_aspects(q)
        whole = [r["page_id"] for r in bm25.rank(q, top_k=limit)]
        rankings = [bm25.rank(a, top_k=top_k) for a in aspects]
        split_ids = interleave_unique(rankings, limit=limit)
        refs = probe["reference_pages"]
        results.append({
            "probe": probe["id"],
            "question": q,
            "aspects": aspects,
            "reference_pages": refs,
            "bm25_full": {"ids": whole, **coverage(whole, refs)},
            "bm25_aspects": {"ids": split_ids, **coverage(split_ids, refs)},
            "note": "Candidate coverage only. No answer-generation, token-budget or correctness claims.",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Existing exploratory baseline JSONL (read only)")
    parser.add_argument("--top-k", type=int, default=10, help="BM25 candidate count per aspect")
    parser.add_argument("--candidate-limit", type=int, default=12, help="Maximum IDs for each comparison")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_limit < 1 or not 1 <= args.max_cases <= 6:
        parser.error("Invalid count argument")
    if args.output.exists() and args.output.stat().st_size:
        parser.error(f"Output already exists: {args.output}; specify a new path")

    # Use ONLY the independent development probes; do not open questions.json.
    from app.retrieval import load_pages
    from scripts.day5_development_probes import load_probes, validate_probes
    pages = load_pages()
    probes = load_probes()[:args.max_cases]
    validate_probes(probes, {p["page_id"] for p in pages})
    results = diagnose(pages, probes, top_k=args.top_k, limit=args.candidate_limit)

    # Existing API-backed results are used only for after-the-fact comparison.
    if args.input.is_file():
        with args.input.open(encoding="utf-8") as f:
            baseline = {r["probe"]: r for line in f if line.strip() for r in [json.loads(line)]}
        for record in results:
            if record["probe"] not in baseline:
                continue
            existing = baseline[record["probe"]]["methods"]
            existing_pool = set(existing["vector"]["ids"]) | set(existing["graph"]["ids"])
            missing = set(record["reference_pages"]) - existing_pool
            record["prior_pool_missing_reference_pages"] = sorted(missing)
            record["bm25_full_new_reference_pages"] = sorted(missing & set(record["bm25_full"]["ids"]))
            record["bm25_aspects_new_reference_pages"] = sorted(missing & set(record["bm25_aspects"]["ids"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    for record in results:
        a, b = record["bm25_full"], record["bm25_aspects"]
        print(f"{record['probe']}: full={a['hits']}/{a['total']}  aspects={b['hits']}/{b['total']}  "
              f"parts={len(record['aspects'])}")
        print(f"  Missing full: {a['missing']}; missing aspects: {b['missing']}")
        if "prior_pool_missing_reference_pages" in record:
            print("  Prior Vector∪Graph missed:", record["prior_pool_missing_reference_pages"])
            print("  BM25 full newly found:", record["bm25_full_new_reference_pages"])
            print("  BM25 aspects newly found:", record["bm25_aspects_new_reference_pages"])
    print(f"Report: {args.output}")
    print("NOTE: No embedding/generation API calls. Gold page IDs were used only post-retrieval.")


if __name__ == "__main__":
    main()
