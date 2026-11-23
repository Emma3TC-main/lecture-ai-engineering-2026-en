"""Lecture 05: offline, development-only vector/lexical fusion ablation.

Read existing retrieval/probe JSONL artifacts. Do not import or inspect the holdout,
invoke model APIs, or adjust selected pages using annotated references.

Usage:
  uv run python -m scripts.day5_fusion_ablation \
    --baseline .day5_benchmark_outputs/dev_probes_full.jsonl \
    --lexical .day5_benchmark_outputs/candidate_probe_v1.jsonl \
    --output .day5_benchmark_outputs/fusion_ablation_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / '.day5_benchmark_outputs/dev_probes_full.jsonl'
DEFAULT_LEXICAL = ROOT / '.day5_benchmark_outputs/candidate_probe_v1.jsonl'
DEFAULT_OUTPUT = ROOT / '.day5_benchmark_outputs/fusion_ablation_v1.jsonl'


def read_unique_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, dict] = {}
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        pid = record['probe']
        if pid in records:
            raise ValueError(f'Duplicate probe {pid} in {path}:{lineno}')
        records[pid] = record
    return records


def rrf_ranks(
    ranked: dict[str, list[str]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[str]:
    """RRF from ranks only. Gold/reference pages never enter this function."""
    if rrf_k < 1:
        raise ValueError('rrf_k must be positive')
    weights = weights or {source: 1.0 for source in ranked}
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError('weights must be positive')
    scores: dict[str, float] = defaultdict(float)
    appearances: dict[str, int] = defaultdict(int)
    first_rank: dict[str, int] = {}
    for source, ids in ranked.items():
        if source not in weights:
            raise ValueError(f'Missing weight for {source}')
        for i, pid in enumerate(dict.fromkeys(ids), 1):
            scores[pid] += weights[source] / (rrf_k + i)
            appearances[pid] += 1
            first_rank[pid] = min(first_rank.get(pid, i), i)
    return sorted(scores, key=lambda pid: (-scores[pid], -appearances[pid], first_rank[pid], pid))


def select_budget(
    ids: list[str],
    pages: dict[str, dict],
    *,
    max_pages: int = 4,
    max_content_chars: int = 5000,
) -> list[str]:
    """Greedy budget selection. Uses no gold labels or reference IDs."""
    if max_pages < 1 or max_content_chars < 1:
        raise ValueError('Budget must be positive')
    chosen: list[str] = []
    used = 0
    for pid in dict.fromkeys(ids):
        if pid not in pages:
            raise ValueError(f'Unknown candidate page: {pid}')
        size = len(pages[pid]['content'])
        if size and used + size <= max_content_chars:
            chosen.append(pid)
            used += size
        if len(chosen) >= max_pages:
            break
    return chosen


def coverage(ids: list[str], references: list[str]) -> dict:
    found = set(ids) & set(references)
    return {'hits': len(found), 'total': len(references), 'missing': sorted(set(references) - set(ids))}


def evaluate_pair(base: dict, lex: dict, pages: dict[str, dict]) -> dict:
    if base['probe'] != lex['probe'] or base['question'] != lex['question']:
        raise ValueError('Probe/question mismatch between evidence files')
    methods = base['methods']
    q = base['question']
    vector = methods['vector']['ids']
    graph = methods['graph']['ids']
    full = lex['bm25_full']['ids']
    aspects = lex['bm25_aspects']['ids']
    pool = list(dict.fromkeys(vector + graph + full + aspects))
    # Policies fixed before evaluation, not chosen based on ground truth.
    options = {
        'vector_budget': methods['vector_budget']['ids'],
        'hybrid_v3a': methods['hybrid']['ids'],
        'rrf_vector_full': select_budget(rrf_ranks({'vector': vector, 'bm25_full': full}), pages),
        'rrf_vector_full_aspects': select_budget(rrf_ranks({'vector': vector, 'bm25_full': full, 'bm25_aspects': aspects}), pages),
        'rrf_vector_graph_full_aspects': select_budget(rrf_ranks({'vector': vector, 'graph': graph, 'bm25_full': full, 'bm25_aspects': aspects}), pages),
    }
    # Quality annotations read only *after* all policies and pages are selected.
    refs = base['provisional_reference_pages']
    if refs != lex['reference_pages']:
        raise ValueError('Ground-truth annotation mismatch')
    return {
        'probe': base['probe'], 'question': q,
        'methods': {key: {'ids': ids, 'pages': len(ids),
                          'chars': sum(len(pages[pid]['content']) for pid in ids),
                          **coverage(ids, refs)}
                    for key, ids in options.items()},
        'candidate_union': {'pages': len(pool), **coverage(pool, refs)},
        'source_candidates': {'vector': len(vector), 'graph': len(graph),
                              'bm25_full': len(full), 'bm25_aspects': len(aspects)},
        'evaluation_note': 'Reference annotations only used after ranking and budget selection; coverage is not answer accuracy.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--lexical', type=Path, default=DEFAULT_LEXICAL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        parser.error(f'Output already exists: {args.output}. Use a different output path.')
    from app.retrieval import load_pages
    corpus = {page['page_id']: page for page in load_pages()}
    base, lexical = read_unique_jsonl(args.baseline), read_unique_jsonl(args.lexical)
    if base.keys() != lexical.keys():
        parser.error('Probe IDs do not match across sources')
    if not base or not all(pid.startswith('P') for pid in base):
        parser.error('Only development probe IDs P* are allowed; holdout is excluded')
    results = [evaluate_pair(base[pid], lexical[pid], corpus) for pid in sorted(base)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as out:
        for record in results:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        print(f"\n{r['probe']} candidate union: {r['candidate_union']['hits']}/{r['candidate_union']['total']}")
        for name, stats in r['methods'].items():
            totals[name][0] += stats['hits']; totals[name][1] += stats['total']
            print(f"  {name:32s} {stats['hits']}/{stats['total']} pages={stats['pages']} chars={stats['chars']} missed={stats['missing']}")
    print('\n=== DEVELOPMENT-ONLY TOTALS ===')
    for name, (hits, count) in totals.items():
        print(f'{name:32s}: {hits}/{count}')
    print('No embeddings or generation calls. Do not interpret reference recall as answer correctness.')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
