# Geometry Analysis: Corrected Findings

## BOS Outlier Bug (Fixed)

The initial analysis claimed "1D collapse by layer 2" — this was caused by position 0 (BOS token) having hidden state norm ~5,126 while all other tokens were ~27. This single outlier dominated PCA, producing a spurious effective rank of 1. All results below exclude position 0.

## Corrected Summary (Layer 8)

| Condition | Eff Rank | PC1% | Speed | Tortuosity | Low Freq% | β₁ |
|-----------|---------|------|-------|------------|-----------|-----|
| text@PT | 35.5 | 5.4% | 23.8 | 384 | 21.4% | 27.4 |
| ts@FT | 13.7 | 38.3% | 91.2 | 400 | 42.9% | 6.8 |
| rand@PT | 43.4 | 6.7% | 26.1 | 437 | 11.9% | 11.9 |
| rand@RandomInit | 13.1 | 29.8% | 25.4 | 199 | 47.5% | 39.5 |

## Finding 1: Pretrained models produce HIGHER-dimensional trajectories

- text@PT: eff_rank=35.5, rand@PT: eff_rank=43.4
- ts@FT: eff_rank=13.7, rand@RandomInit: eff_rank=13.1
- Pretraining creates richer, higher-dimensional trajectories
- FT and RandomInit are more concentrated (fewer effective dimensions)
- PC1 captures only 5-7% of variance for PT models vs 30-38% for FT/RandomInit

## Finding 2: ts@FT has distinct spectral properties — lower frequency, higher speed

- ts@FT: 42.9% low-frequency energy vs text@PT: 21.4%, rand@PT: 11.9%
- ts@FT trajectory speed is 3.8x higher than text@PT (91 vs 24)
- The FT model makes larger, smoother updates per timestep when processing TS
- This is direct evidence that finetuning reshapes trajectory dynamics toward temporal patterns

## Finding 3: ts@FT is topologically simpler (fewer loops)

- ts@FT: β₁ ≈ 6.8 (few loops)
- text@PT: β₁ ≈ 27.4 (many loops)
- rand@RandomInit: β₁ ≈ 39.5 (most loops)
- TS processing produces smoother, more linear trajectories
- Text processing creates more complex topology (sentence/phrase structures?)

## Finding 4: Text spreads 3-7x more along PC1 than random tokens (layers 4-20)

- Peak at L4: 7.4x ratio
- Mid layers (L8-L16): 1.8-4.4x ratio
- At embedding (L0-L2): random has MORE spread (0.3x) — wider vocabulary coverage
- At L27: equal (1.0x) — final layer differentiates all inputs for next-token prediction
- Cross-sequence correlations are near-zero for both — sequences are genuinely different
- Text's higher spread along the dominant PC direction explains its greater TS diversity when linearly decoded

## Finding 5: Subspace alignment

- text@PT and rand@PT span the most similar subspaces (~68° mean principal angle) — same model, same weight-determined geometry
- text@PT vs ts@FT is less aligned (~73°) — finetuning rotates the subspace
- All vs rand@RandomInit is most different (~75-85°)

## Mechanistic Story (Corrected)

1. Pretraining creates **high-dimensional** representation trajectories (eff_rank ~35-43), NOT 1D collapse
2. TS finetuning reshapes trajectories toward **lower dimensionality** (eff_rank ~14), **lower frequency**, and **higher speed** — properties matching time series structure
3. Text inputs produce **more spread** across the dominant directions (3-7x vs random), enabling diverse linear projections into TS space
4. The key difference between text and random is NOT manifold shape (both have eff_rank ~35-43 through PT) but **how widely different inputs explore** the manifold
5. RandomInit produces trajectories with similar dimensionality to FT (~13) but different spectral properties (high-frequency, many loops) — raw architecture provides moderate compression but not the right temporal structure
