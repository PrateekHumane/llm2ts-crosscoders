"""
Search crosscoder features for WikiText passages where words repeat.
For each layer's PT_FT features, check wiki_spans.json for passages
containing repeated words near the activation peak.
"""
import json
import re
from pathlib import Path
from collections import Counter

ANALYSIS = Path('/workspace/llm2ts-crosscoders/analysis')
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'not', 'no', 'nor',
    'so', 'if', 'then', 'than', 'that', 'this', 'these', 'those', 'it',
    'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'we', 'our',
    'you', 'your', 'my', 'me', 'him', 'who', 'which', 'what', 'where',
    'when', 'how', 'as', 'up', 'out', 'about', 'into', 'over', 'after',
    'before', 'between', 'under', 'during', 'through', 'also', 'more',
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'very',
    'just', 'because', 'each', 'all', 'both', 'few', 'many', 'much',
    'any', 'every', 'while', 'since', 'until', 'although', 'though',
    'however', 'there', 'here', 'still', 'yet', 'already', 'even',
    'well', 'back', 'one', 'two', 'three', 'first', 'new', 'now',
    'way', 'time', 'year', 'years', 'part', 'made', 'make', 'like',
    'long', 'get', 'got', 'come', 'came', 'go', 'went', 'see', 'saw',
    'take', 'took', 'know', 'knew', 'think', 'thought', 'say', 'said',
    'use', 'used', 'find', 'found', 'give', 'gave', 'tell', 'told',
    'work', 'worked', 'call', 'called', 'try', 'tried', 'ask', 'asked',
    'need', 'needed', 'feel', 'felt', 'become', 'became', 'leave', 'left',
    'put', 'mean', 'keep', 'let', 'begin', 'began', 'seem', 'seemed',
    'help', 'show', 'showed', 'hear', 'heard', 'play', 'played', 'run',
    'move', 'moved', 'live', 'lived', 'believe', 'hold', 'bring', 'brought',
    'happen', 'write', 'written', 'provide', 'sit', 'stand', 'lose', 'lost',
    'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change',
    'lead', 'led', 'close', 'turn', 'start', 'started', 'number',
    'point', 'end', 'day', 'days', 'life', 'hand', 'high', 'last',
    'large', 'great', 'small', 'old', 'young', 'different', 'important',
    'unk', 'also', 'later', 'early', 'second', 'film', 'season',
}


def find_repeated_words(text, min_repeats=3, min_word_len=3):
    """Find non-stopword words that appear >= min_repeats times."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    words = [w for w in words if len(w) >= min_word_len and w not in STOPWORDS]
    counts = Counter(words)
    return {w: c for w, c in counts.items() if c >= min_repeats}


def search_layer(layer_num):
    layer_dir = ANALYSIS / f'layer_{layer_num}' / 'PT_FT'
    ranking_file = layer_dir / 'ranking.json'
    if not ranking_file.exists():
        return []

    ranking = json.load(open(ranking_file))
    results = []

    for entry in ranking:
        fid = entry['feature_id']
        wiki_file = layer_dir / f'feature_{fid}' / 'wiki_spans.json'
        if not wiki_file.exists():
            continue

        spans = json.load(open(wiki_file))
        feature_rep_score = 0
        rep_examples = []

        for span in spans:
            text = span.get('full_text', '')
            repeats = find_repeated_words(text, min_repeats=3, min_word_len=3)
            if repeats:
                score = sum(repeats.values())
                feature_rep_score += score
                rep_examples.append({
                    'act': span['activation_value'],
                    'repeats': repeats,
                    'text_snippet': text[:200]
                })

        if feature_rep_score > 0:
            results.append({
                'layer': layer_num,
                'feature_id': fid,
                'rate_pt': entry.get('rate_pt', 0),
                'rate_ft': entry.get('rate_ft', 0),
                'rate_wiki': entry.get('rate_wiki', 0),
                'rep_score': feature_rep_score,
                'n_spans_with_repeats': len(rep_examples),
                'examples': rep_examples[:3]
            })

    results.sort(key=lambda x: x['rep_score'], reverse=True)
    return results


if __name__ == '__main__':
    import sys
    layers = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(0, 28))

    all_results = []
    for layer in layers:
        layer_dir = ANALYSIS / f'layer_{layer}' / 'PT_FT'
        if not (layer_dir / 'ranking.json').exists():
            continue
        results = search_layer(layer)
        all_results.extend(results)
        if results:
            top = results[0]
            print(f"Layer {layer}: best rep_score={top['rep_score']} "
                  f"(F{top['feature_id']}, {top['n_spans_with_repeats']}/10 spans)")
        else:
            print(f"Layer {layer}: no features with repeated words")

    print(f"\n{'='*70}")
    print(f"TOP 20 FEATURES BY WORD REPETITION SCORE")
    print(f"{'='*70}")
    all_results.sort(key=lambda x: x['rep_score'], reverse=True)
    for r in all_results[:20]:
        print(f"\nLayer {r['layer']}, Feature {r['feature_id']} | "
              f"PT={r['rate_pt']:.1%} FT={r['rate_ft']:.1%} Wiki={r['rate_wiki']:.1%} | "
              f"rep_score={r['rep_score']} ({r['n_spans_with_repeats']}/10 spans)")
        for ex in r['examples'][:2]:
            reps = ', '.join(f'{w}({c})' for w, c in sorted(
                ex['repeats'].items(), key=lambda x: -x[1]))
            print(f"  act={ex['act']:.1f} repeats: {reps}")
            print(f"  \"{ex['text_snippet'][:120]}...\"")
