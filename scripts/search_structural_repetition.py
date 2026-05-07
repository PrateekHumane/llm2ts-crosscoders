"""
Search for features where WikiText passages contain STRUCTURAL repetition:
- Consecutive identical words/tokens ("the the", "3 3 3")
- Repeated short phrases ("and then... and then... and then")
- Repeated number patterns ("2-3-2", "1,2,1,2")
"""
import json
import re
from pathlib import Path

ANALYSIS = Path('/workspace/llm2ts-crosscoders/analysis')


def structural_repetition_score(text):
    """Score how much structural repetition a text contains."""
    score = 0
    details = []

    words = text.lower().split()

    # 1. Consecutive identical words: "the the", "3 3 3"
    streak = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            streak += 1
        else:
            if streak >= 2:
                score += streak * 2
                details.append(f'"{words[i-1]}"x{streak}')
            streak = 1
    if streak >= 2:
        score += streak * 2
        details.append(f'"{words[-1]}"x{streak}')

    # 2. Repeated number-dash patterns: "2-3-2", "3·3·3"
    num_patterns = re.findall(r'(\d[\s\-–—·×,]+){2,}\d', text)
    for p in num_patterns:
        score += 5
        details.append(f'num_pattern: "{p.strip()}"')

    # 3. Repeated short phrases (2-4 word ngrams appearing 3+ times)
    for n in [2, 3, 4]:
        if len(words) < n:
            continue
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        from collections import Counter
        counts = Counter(ngrams)
        for ng, cnt in counts.items():
            if cnt >= 3:
                score += cnt * n
                details.append(f'phrase "{ng}"x{cnt}')

    # 4. Repeated single characters/symbols: "= = =", "- - -"
    sym_patterns = re.findall(r'([=\-@#*]{2,}(?:\s+[=\-@#*]{2,})+)', text)
    for p in sym_patterns:
        score += 3
        details.append(f'sym: "{p[:30]}"')

    return score, details


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
        total_score = 0
        span_scores = []

        for span in spans:
            text = span.get('full_text', '')
            s, d = structural_repetition_score(text)
            total_score += s
            if s > 0:
                span_scores.append({
                    'act': span['activation_value'],
                    'score': s,
                    'details': d[:5],
                    'text': text[:200]
                })

        if total_score > 20:
            results.append({
                'layer': layer_num,
                'feature_id': fid,
                'rate_pt': entry.get('rate_pt', 0),
                'rate_ft': entry.get('rate_ft', 0),
                'rate_wiki': entry.get('rate_wiki', 0),
                'total_score': total_score,
                'n_spans': len(span_scores),
                'examples': sorted(span_scores, key=lambda x: -x['score'])[:3]
            })

    results.sort(key=lambda x: x['total_score'], reverse=True)
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
            print(f"Layer {layer}: best structural_rep={top['total_score']} "
                  f"(F{top['feature_id']}, {top['n_spans']}/10 spans)")
        else:
            print(f"Layer {layer}: no structural repetition features")

    print(f"\n{'='*70}")
    print(f"TOP 15 FEATURES BY STRUCTURAL REPETITION")
    print(f"{'='*70}")
    all_results.sort(key=lambda x: x['total_score'], reverse=True)
    for r in all_results[:15]:
        print(f"\nLayer {r['layer']}, Feature {r['feature_id']} | "
              f"PT={r['rate_pt']:.1%} FT={r['rate_ft']:.1%} Wiki={r['rate_wiki']:.1%} | "
              f"struct_score={r['total_score']} ({r['n_spans']}/10 spans)")
        for ex in r['examples'][:2]:
            print(f"  score={ex['score']} details: {ex['details']}")
            print(f"  \"{ex['text'][:150]}...\"")
