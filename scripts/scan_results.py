"""Quick scan of PT_FT analysis results across all layers."""
import json, os, sys

ANALYSIS_DIR = "analysis"

for layer_idx in sorted([int(d.split("_")[1]) for d in os.listdir(ANALYSIS_DIR) if d.startswith("layer_")]):
    layer_dir = os.path.join(ANALYSIS_DIR, f"layer_{layer_idx}", "PT_FT")
    ranking_path = os.path.join(layer_dir, "ranking.json")
    if not os.path.exists(ranking_path):
        continue
    
    with open(ranking_path) as f:
        ranking = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx} — {len(ranking)} PT_FT features ranked")
    print(f"{'='*70}")
    
    for r in ranking[:10]:
        fid = r["feature_id"]
        rate_pt = r.get("rate_pt", 0) * 100
        rate_ft = r.get("rate_ft", 0) * 100
        rate_wiki = r.get("rate_wiki", 0) * 100
        balance = min(rate_pt, rate_ft)
        
        # Check for qualitative score
        info_path = os.path.join(layer_dir, f"feature_{fid}", "info.json")
        score = None
        interp = ""
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            score = info.get("qualitative_score")
            interp = (info.get("interpretation") or "")[:80]
        
        score_str = f"[{score}]" if score else "[--]"
        wiki_str = f"wiki={rate_wiki:.1f}%" if rate_wiki > 0 else ""
        print(f"  #{r.get('rank', '?'):2} feat={fid:4d}  PT={rate_pt:.1f}%  FT={rate_ft:.1f}%  "
              f"bal={balance:.1f}%  {wiki_str}  {score_str} {interp}")
