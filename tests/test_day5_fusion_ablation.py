"""Deterministic, API-free testing of development-only fusion ablations."""
import pytest
from scripts.day5_fusion_ablation import coverage, evaluate_pair, read_unique_jsonl, rrf_ranks, select_budget


def page(pid, length=10):
    return {'page_id': pid, 'content': 'z' * length}


def test_rrf_combines_multiple_rankings():
    ranked = rrf_ranks({'vector': ['a', 'b'], 'bm25': ['b', 'c']})
    assert ranked[0] == 'b'
    assert set(ranked) == {'a', 'b', 'c'}


def test_rrf_deduplicates_within_source():
    assert rrf_ranks({'one': ['a', 'a', 'b'], 'two': ['b']})[0] == 'b'


def test_rrf_rejects_invalid_settings():
    with pytest.raises(ValueError):
        rrf_ranks({'a': ['x']}, rrf_k=0)
    with pytest.raises(ValueError):
        rrf_ranks({'a': ['x']}, weights={'a': 0})


def test_budget_dedup_and_both_limits():
    corpus = {x: page(x, 6) for x in ['a', 'b', 'c']}
    assert select_budget(['a', 'a', 'b', 'c'], corpus, max_pages=2, max_content_chars=12) == ['a', 'b']
    assert select_budget(['a', 'b', 'c'], corpus, max_pages=3, max_content_chars=7) == ['a']


def test_budget_skips_large_page():
    assert select_budget(['large', 'small'], {'large': page('large', 20), 'small': page('small', 4)}, max_content_chars=10) == ['small']


def test_selection_never_reads_reference_pages():
    corpus = {'x': page('x'), 'y': page('y'), 'z': page('z')}
    methods = {'vector': {'ids': ['x', 'y']}, 'graph': {'ids': ['y', 'z']},
               'vector_budget': {'ids': ['x', 'y']}, 'hybrid': {'ids': ['x', 'y']}}
    base = {'probe': 'P01', 'question': 'What about cats and dogs?',
            'methods': methods, 'provisional_reference_pages': ['z']}
    lexical = {'probe': 'P01', 'question': base['question'], 'reference_pages': ['z'],
               'bm25_full': {'ids': ['z', 'y']}, 'bm25_aspects': {'ids': ['y', 'z']}}
    a = evaluate_pair(base, lexical, corpus)
    base['provisional_reference_pages'] = ['x']
    lexical['reference_pages'] = ['x']
    b = evaluate_pair(base, lexical, corpus)
    assert {key: val['ids'] for key, val in a['methods'].items()} == {key: val['ids'] for key, val in b['methods'].items()}
    assert a['candidate_union']['hits'] == b['candidate_union']['hits'] == 1


def test_reference_coverage_is_posthoc():
    assert coverage(['a'], ['a', 'b']) == {'hits': 1, 'total': 2, 'missing': ['b']}
