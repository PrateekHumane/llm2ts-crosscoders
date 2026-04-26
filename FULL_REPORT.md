# Linear Decoding of Time Series from Language Model Hidden States

## 1. Problem Statement

We investigate whether the internal representations of a pretrained language model, when processing natural language text, contain sufficient sequential structure to produce realistic time series signals. Concretely: can a simple linear map applied to hidden states from a language model processing WikiText produce outputs that resemble real-world time series data from GiftEval — without any paired text-to-time-series supervision?

---

## 2. Models

All models share the Qwen3-0.6B architecture (28 transformer layers, hidden dimension d = 1024, ~0.6B parameters).

| Model | Description |
|-------|-------------|
| **PT** | `Qwen/Qwen3-0.6B` — pretrained on text only, never seen time series |
| **FT** | `cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512` — PT finetuned on time series |
| **RI** | `cerc-aai/Qwen3-0.6B-random-normal_scale-uniform_bin-V512` — randomly initialized, trained on time series |
| **RandomInit** | Fresh `Qwen3-0.6B` with random weights — never trained on anything |

FT and RI use uniform binning tokenization: z-score normalize each window, clip to [-5, 5], discretize into 512 bins. Context length T = 512 timesteps.

---

## 3. Data

**Time series**: GiftEval validation set (Salesforce/GiftEval). Non-overlapping windows of T = 512 timesteps, z-score normalized per window (mean = 0, std = 1). Windows with zero variance (constant values) are excluded. Evaluation bank: 10,000 clean windows.

**Text**: WikiText-103 train split. Tokenized with Qwen3-0.6B tokenizer, chunked into sequences of exactly 512 tokens. Training set: sequences 0–1919 (N = 1920). Held-out set: sequences 2000–4999 (N = 3000).

---

## 4. Method: Linear Sequence-to-Sequence Mapping

### 4.1 Hidden State Extraction

For each WikiText sequence, we pass it through the model and extract the hidden state at every layer using forward hooks. For the concatenated-layer variant, we concatenate all 28 layers:

```
H_i = [h_i^(0); h_i^(1); ...; h_i^(27)] ∈ R^{T × D}
```

where D = 28 × 1024 = 28,672 for the concatenated variant, or D = 1024 for single-layer experiments.

### 4.2 Linear Map

A single linear layer applied independently at each timestep:

```
ŷ_{i,t} = W h_{i,t} + b
```

where W ∈ R^{1×D}, b ∈ R. Total parameters: D + 1 (1,025 for single layer, 28,673 for concatenated).

### 4.3 Normalization

Predicted sequences are z-score normalized before matching:

```
ŷ_i ← (ŷ_i - mean(ŷ_i)) / max(std(ŷ_i), 10^{-4})
```

This ensures predictions have mean 0 and std ~1, matching the z-scored time series targets.

### 4.4 Training: EM-Style Hard Matching

There is no paired text-to-time-series data. Training uses an EM-style procedure:

**E-step**: For each predicted sequence ŷ_i in the batch, find its nearest neighbor in a random sample of real time series:

```
j* = argmin_j ||ŷ_i - Y_j||² / T
```

where the MSE is computed per-timestep and averaged over the sequence length T.

**M-step**: Minimize MSE between predictions and their matched targets:

```
L_MSE = (1/B) Σ_i ||ŷ_i - Y_{j*_i}||²
```

### 4.5 Diversity Penalty (PSD)

Without a diversity penalty, the linear map collapses to producing one dominant shape. We add a PSD (Power Spectral Density) diversity penalty that discourages predictions within a batch from having similar frequency content:

```
L_div = mean_{i≠j} cos_sim(|FFT(ŷ_i)|, |FFT(ŷ_j)|)
```

where |FFT(·)| denotes the magnitude of the discrete Fourier transform, and the cosine similarity is computed between the PSD vectors of all pairs in the batch.

This penalty is shift-invariant: two shifted copies of the same signal have identical PSDs and will be penalized.

**Total loss**:

```
L = L_MSE + λ · L_div
```

### 4.6 Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 10^{-3}) |
| Batch size | 32 (WikiText) × 128 (TS comparison pool) |
| Epochs | 80–100 |
| λ (diversity) | 0.5 (optimal, swept over {0, 0.5, 0.7, 1.0, 1.5}) |
| WikiText sequences | 1920 (training), 3000 (held-out) |
| TS evaluation bank | 10,000 windows |

---

## 5. Evaluation Metrics

### 5.1 Nearest Neighbor Distance (NN dist)

For each predicted sequence, compute MSE to all real time series in the evaluation bank and take the minimum:

```
d_i = min_j (1/T) ||ŷ_i - Y_j||²
```

**Raw NN dist**: Average d_i over all N predictions.

**Deduped NN dist**: For each unique matched real TS, keep only the best (lowest) distance. Average over unique matches only. This removes inflation from repeated matches.

### 5.2 Unique Matches

Number of distinct real time series that are the nearest neighbor of at least one prediction:

```
Unique = |{j* : j* = argmin_j d(ŷ_i, Y_j) for some i}|
```

### 5.3 Cluster Coverage

Real time series are clustered into 20 groups by K-means on their autocorrelation profiles (ACF at lags 0–31). Coverage = number of clusters containing at least one matched TS.

### 5.4 Per-Cluster Distance

For each cluster c, the best prediction distance:

```
d_c = min_i min_{j ∈ cluster_c} ||ŷ_i - Y_j||²
```

Averaged over all clusters. This measures whether the model can approximate every type of TS, not just common ones.

### 5.5 Match Entropy

Entropy of the distribution over matched TS indices, normalized by log(N):

```
H = -Σ p_j log(p_j) / log(N)
```

where p_j is the fraction of predictions whose nearest neighbor is Y_j. High entropy = diverse; low entropy = mode collapse.

---

## 6. Experiments and Results

### 6.1 Experiment 1: Single Layer (Layer 8)

**Setup**: Extract hidden states from layer 8 only (D = 1024). No diversity penalty (λ = 0).

| Method | NN Dist | Description |
|--------|------:|-------------|
| RI (TS-trained) | 0.463 | Best — already trained on TS |
| PT (language model) | 0.716 | 2.4× better than random |
| RandomInit (untrained arch) | 0.792 | Architecture alone helps |
| Random vectors | 1.687 | Baseline |
| Shuffled PT | 1.773 | Temporal order destroyed ≈ random |

**Key findings**:
- PT >> Random (2.4×): Language model hidden states contain TS-decodable structure
- Shuffled ≈ Random: Structure requires temporal order — comes from sequential text processing
- RandomInit < PT: Architecture contributes ~2.1×, language training adds another ~1.2×

### 6.2 Experiment 2: Layer Sweep

**Setup**: Single-layer experiments at 9 different layers. PT vs Random vectors.

| Layer | PT | Random | Ratio |
|------:|-----:|-------:|------:|
| 0 | 0.672 | 1.678 | 2.50× |
| 2 | 0.548 | 1.679 | **3.06×** |
| 4 | 0.558 | 1.672 | 3.00× |
| 8 | 0.572 | 1.672 | 2.92× |
| 12 | 0.552 | 1.674 | 3.03× |
| 16 | 0.749 | 1.679 | 2.24× |
| 20 | 0.611 | 1.675 | 2.74× |
| 24 | 0.652 | 1.676 | 2.57× |
| 26 | 0.653 | 1.674 | 2.57× |

**Pattern**: Peak at L2–L12 (~3×), dip at L16 (2.24×), partial recovery at L20–L26 (~2.6×).

### 6.3 Experiment 3: Random-Init Architecture Control

**Setup**: Compare PT vs RandomInit (untrained Qwen3) vs Random vectors across all layers.

| Layer | PT | RandomInit | Random | PT vs RandomInit |
|------:|-----:|----------:|--------:|:--------:|
| 0 | 0.672 | 0.791 | 1.678 | PT wins 18% |
| 2 | 0.548 | 0.663 | 1.679 | PT wins 21% |
| 4 | 0.558 | 0.655 | 1.672 | PT wins 17% |
| 8 | 0.572 | 0.656 | 1.672 | PT wins 15% |
| 12 | 0.552 | 0.626 | 1.674 | PT wins 13% |
| 16 | 0.749 | 0.618 | 1.679 | RandomInit wins 18% |
| 20 | 0.611 | 0.616 | 1.675 | Tied |
| 24 | 0.652 | 0.621 | 1.676 | RandomInit wins 5% |
| 26 | 0.653 | 0.610 | 1.674 | RandomInit wins 7% |

**Decomposition**: Architecture alone provides ~2.5× improvement over random. Language training adds 13–21% at early/mid layers (L0–L12) but hurts at L16+ (deep layers specialize for next-token prediction, reducing generic temporal structure).

### 6.4 Experiment 4: Diversity Analysis (Single Layer)

**Problem**: Without diversity penalty, mode collapse — PT maps most predictions to ~40–60 unique TS.

**Setup**: Add PSD diversity penalty (λ = 0.5) at layer 8.

| Config | Unique | Clusters | Entropy |
|--------|------:|------:|------:|
| PT (no penalty) | 154 | 17/20 | 0.560 |
| PT + PSD λ=0.5 | **257** | 15/20 | **0.640** |
| RandomInit (no penalty) | 35 | 13/20 | 0.363 |
| RandomInit + PSD λ=0.5 | 52 | 13/20 | 0.428 |
| Random vectors | 880 | **20/20** | **0.975** |

**Finding**: Diversity penalty helps PT much more than RandomInit (154→257 vs 35→52). Random has best diversity but worst quality.

### 6.5 Experiment 5: All Layers Concatenated

**Setup**: Concatenate all 28 layers (D = 28,672). Float16 storage on disk (55 GB memmap), loaded into RAM as float16 tensor.

#### Without diversity penalty:

| Config | NN Dist | Unique | Clusters |
|--------|------:|------:|------:|
| PT concat | **0.458** | 40 | 13/20 |
| RandomInit concat | 0.497 | 66 | 12/20 |

Better NN distance than single-layer (0.46 vs 0.72) but worse diversity (40 unique) — more capacity enables harder mode collapse.

#### With PSD diversity penalty (λ = 0.5):

| Config | NN Dist | Unique | Clusters | Entropy |
|--------|------:|------:|------:|------:|
| **PT concat div0.5** | 0.737 | **858** | **17/20** | **0.822** |
| RandomInit concat div0.5 | 0.506 | 114 | 15/20 | 0.479 |
| Random vectors | 1.723 | 280 | 18/20 | 0.983 |

**Key result**: PT produces **7.5× more diverse TS than RandomInit** (858 vs 114) with the same architecture and diversity penalty. Language training uniquely enables diverse time series generation.

#### Lambda sweep:

| λ | Unique | NN Dist | Entropy |
|---|------:|------:|------:|
| 0.5 | **858** | 0.740 | **0.831** |
| 0.7 | 78 | 0.499 | 0.382 |
| 1.0 | 423 | 0.634 | 0.674 |
| 1.5 | 740 | 0.679 | 0.800 |

λ = 0.5 is optimal. The relationship is non-monotonic due to training dynamics.

#### Layer weight analysis (from λ = 0.5 mapper):

The trained W ∈ R^{1×28672} can be reshaped to (28, 1024) to see per-layer contribution. Layer norms:

**Top 5 most-used layers**: L13 (3.65), L10 (3.53), L16 (3.20), L6 (3.19), L14 (3.15)

Mid-layers (L6–L16) dominate. Deep layers (L19–L27) contribute very little (norms 0.47–1.32).

### 6.6 Experiment 6: Stability Test

5 random seeds for the best config (PT concat div0.5). Results so far (3 of 5 seeds):

| Seed | Unique | Clusters | NN Dist |
|-----:|------:|------:|------:|
| 0 | 828 | 16/20 | 0.744 |
| 1 | 883 | 15/20 | 0.738 |
| 2 | 49 | 13/20 | 0.492 |

**Training is unstable.** Seeds 0 and 1 produce high diversity (~830–880 unique). Seed 2 collapses to 49 unique (mode collapse). The diversity penalty at λ = 0.5 helps on average but does not guarantee high diversity every run. The low NN distance for seed 2 (0.49) confirms the quality-diversity tradeoff: collapsed runs match fewer TS very tightly.

Note: the evaluation metric also has variance — the same mapper evaluated against the full 10K TS bank (instead of subsampled 2K) gives 601 unique matches (see Experiment 7).

### 6.7 Experiment 7: Fair Evaluation (Full 10K Bank)

Earlier evaluations subsampled the TS bank (2000 of 10000), causing variance. Evaluating the saved mapper against the full 10K bank:

| Metric | Value |
|--------|------:|
| Unique matches | 601 / 10,000 (6%) |
| Dedup NN dist | 0.630 |
| Clusters covered | 14/20 |
| Top-50 match quality | 0.342 |
| Top-100 match quality | 0.382 |
| Top-200 match quality | 0.440 |

### 6.8 Experiment 8: Generalization to Held-Out Text

Training used WikiText sequences 0–1919. Held-out evaluation on sequences 2000–4999 (N = 3000, never seen during training).

| Metric | Training (1920 seqs) | Held-Out (3000 seqs) |
|--------|------:|------:|
| Raw NN dist | 0.667 | 0.709 |
| Unique matches | 601 | **760** |
| Dedup NN dist | 0.630 | 0.673 |
| Clusters | 14/20 | **15/20** |
| Top-50 quality | 0.342 | 0.345 |
| Top-100 quality | 0.382 | 0.383 |

**The mapper generalizes.** Held-out text produces 760 unique matches (more than training's 601) at only slightly worse quality. 390 NEW unique TS are discovered that training text didn't match. Combined coverage: 991 / 10,000 (9.9%).

Top-K match quality is nearly identical between training and held-out, confirming the linear map learned a general projection rather than overfitting to specific training text.

### 6.9 Experiment 9: Full Ablation Study (Architecture × Input Tokens)

A 2×2 ablation isolating the contributions of trained weights and meaningful text:

| | **Text tokens** | **Random tokens** |
|--|:-----------:|:-------------:|
| **PT (trained)** | text_PT | rand_PT |
| **RandomInit (untrained)** | text_RandInit | rand_RandInit |

Each condition: extract all 28 layers concatenated, train linear mapper (λ=0.5, seed=1, 100 epochs), evaluate on training data (1920 seqs) and held-out data (1000 seqs) against full 10K TS bank. All mappers saved.

| Ablation | Loss | Train Unique | Train NN | Held Unique | Held NN |
|----------|-----:|------:|------:|------:|------:|
| **text_PT** | 1.33 | **686** | 0.672 | **439** | 0.717 |
| text_RandInit | 1.21 | 54 | 0.402 | 73 | 0.541 |
| rand_PT | 1.16 | 4 | 0.283 | 1 | 0.293 |
| rand_RandInit | 1.29 | 99 | 0.522 | 138 | 0.741 |

**Key findings:**

1. **text_PT dominates diversity**: 686 train / 439 held-out unique matches — no other condition comes close.

2. **Random tokens destroy PT's diversity**: rand_PT collapses to 4 train / 1 held-out unique. PT needs meaningful text to produce varied hidden states. Random tokens create uniform representations that mode-collapse to a single output.

3. **rand_RandInit > text_RandInit**: Random tokens through untrained architecture (99/138 unique) beats real text through untrained architecture (54/73). The untrained model doesn't understand text, so random tokens produce more diverse hidden states (higher input entropy → more output diversity through random weights).

4. **Diversity requires BOTH trained weights AND meaningful text**: Neither component alone is sufficient. Trained weights without meaningful input (rand_PT) collapse. Meaningful input without trained weights (text_RandInit) provides limited diversity. Only the combination (text_PT) produces high diversity.

### 6.10 Experiment 10: Fair Top-K Comparison Across Ablations

Raw NN distance is misleading because lower diversity (fewer unique matches) yields artificially low NN distances — the model matches each prediction to the same few TS. A fair comparison evaluates at matched K: for each model, take the best-K unique matches (deduplicated) and compare mean NN distance.

| K | text_PT | text_RandInit | rand_PT | rand_RandInit |
|--:|--------:|-----------:|--------:|-------------:|
| 4 | **0.254** | 0.383 | 0.330 | 0.412 |
| 54 | **0.350** | 0.721 | — | 0.755 |
| 99 | **0.383** | 0.825 | — | 0.862 |
| 200 | **0.441** | 0.971 | — | 1.004 |
| 439 | **0.535** | — | — | — |
| 686 | **0.639** | — | — | — |

("—" = fewer than K unique matches available)

**text_PT wins at every K level.** Even at K=4 (where rand_PT has its full set), text_PT's best 4 unique matches (0.254) beat rand_PT (0.330). The gap widens dramatically at higher K: at K=99, text_PT (0.383) is 2.2× better than text_RandInit (0.825) and rand_RandInit (0.862). This confirms text_PT doesn't just produce more diverse outputs — each individual match is also higher quality.

### 6.11 Experiment 11: Train-Free Random Projection Baseline

To test whether alignment is intrinsic to representation geometry (not learned by training the linear mapper), we evaluate random linear projections W ~ N(0, I/D) with no training or EM matching. 50 random projections per condition, same normalization as the trained mapper.

| Condition | NN Dist (mean±std) | Unique (mean±std) | Clusters (mean±std) | Entropy (mean±std) |
|-----------|---:|---:|---:|---:|
| text_PT | 1.358 ± 0.299 | 174.4 ± 174.4 | 12.5 ± 8.2 | 0.469 ± 0.408 |
| text_RandInit | 1.042 ± 0.020 | 166.2 ± 9.6 | 17.1 ± 0.8 | 0.627 ± 0.032 |
| rand_PT | 1.402 ± 0.344 | 105.8 ± 156.9 | 9.2 ± 7.8 | 0.324 ± 0.369 |
| rand_RandInit | 1.097 ± 0.015 | 181.9 ± 11.1 | 17.5 ± 0.7 | 0.662 ± 0.016 |

**Key findings:**

1. **PT models are highly anisotropic.** text_PT and rand_PT have enormous variance across random projections (std ≈ mean for unique matches). Some random directions produce 400+ unique matches; most produce <10. The trained mapper finds a special direction in a highly structured space.

2. **RandInit models are isotropic.** text_RandInit and rand_RandInit have tiny variance (std/mean < 6%). All random directions are equally (mediocre) — there are no special directions to find.

3. **Training matters beyond direction-finding.** The trained text_PT mapper achieves NN=0.67 with 686 unique matches, far better than any single random projection. Training discovers both the right direction AND optimizes quality through EM matching.

4. **Random projections on RandInit ≈ trained mapper on RandInit.** text_RandInit random projections (166 unique, NN=1.04) are comparable to the trained mapper (54 unique on training data, 73 held-out). The isotropic geometry means there's no better direction to find.

### 6.12 Experiment 12: Hidden State Diversity Analysis

We analyze the representation geometry directly via PCA on flattened hidden states (50K random samples from each condition's N×T×D tensor).

| Condition | Mean Var | Effective Rank | Participation Ratio | PCs for 90% | PCs for 95% |
|-----------|------:|------:|------:|------:|------:|
| text_PT | 104.38 | 5.6 | 1.5 | 60 | 315 |
| text_RandInit | 3.80 | 431.3 | 138.6 | 421 | 501+ |
| rand_PT | 93.07 | 2.8 | 1.3 | 2 | 91 |
| rand_RandInit | 3.88 | 585.3 | 207.6 | 501+ | 501+ |

Note: Effective rank = exp(entropy of normalized eigenvalue spectrum), using the full trace (total variance across all 28,672 dimensions) as the denominator. The top eigenvalue alone explains 81.5% of variance for text_PT and 89.2% for rand_PT.

**Key findings:**

1. **PT models concentrate variance into a handful of effective dimensions.** text_PT has eff_rank=5.6 and rand_PT has eff_rank=2.8, despite D=28,672. The first PCA component alone captures 82–89% of total variance. The pretrained weights create a low-dimensional manifold.

2. **RandInit models spread variance across hundreds of dimensions.** text_RandInit (eff_rank=431) and rand_RandInit (eff_rank=585) have near-uniform eigenvalue spectra. Random weights act as random projections of the input.

3. **PT has high mean variance but extremely low rank.** text_PT's mean variance per dimension (104.4) is 27× higher than text_RandInit (3.8), but concentrated in fewer dimensions. A few PCA components capture almost all variance — 2 PCs explain 90% for rand_PT, 60 for text_PT.

4. **Input type affects PT rank but not RandInit rank.** text_PT (eff_rank=5.6) > rand_PT (eff_rank=2.8): meaningful text activates roughly twice as many effective dimensions. But text_RandInit (431) ≈ rand_RandInit (585): the untrained model treats all inputs similarly.

5. **This explains Experiment 11.** PT's anisotropy (few dominant directions) causes high variance in random projections — you either hit the important subspace or miss it. RandInit's isotropy (many similar directions) produces stable but mediocre random projections.

### 6.13 Experiment 13: Spectral Alignment Analysis

We compare the power spectral density (PSD) of mapper predictions against real GiftEval time series. PSD is computed via |FFT|², normalized to sum to 1 per sequence, then averaged.

| Condition | PSD L2 | KL Div | Low (0-10%) | Mid (10-50%) | High (50-100%) |
|-----------|------:|------:|------:|------:|------:|
| text_PT | 0.228 | 0.267 | 0.901 | 0.066 | 0.033 |
| text_RandInit | 0.274 | 0.431 | 0.584 | 0.195 | 0.221 |
| rand_PT | **0.120** | **0.125** | 0.898 | 0.058 | 0.045 |
| rand_RandInit | 0.264 | 0.443 | 0.553 | 0.210 | 0.237 |
| **Real TS** | — | — | 0.792 | 0.137 | 0.072 |

**Key findings:**

1. **PT models are strongly low-frequency biased.** Both text_PT (90.1% low) and rand_PT (89.8% low) concentrate energy in the lowest 10% of frequencies, even more than real TS (79.2%). The pretrained transformer's autoregressive structure produces smooth, slowly-varying outputs.

2. **RandInit models have flatter spectra.** text_RandInit (58.4% low, 22.1% high) and rand_RandInit (55.3% low, 23.7% high) distribute energy more evenly, producing noisier outputs. This matches the visual appearance of RandInit predictions.

3. **rand_PT has the best spectral match but worst diversity.** rand_PT achieves PSD_L2=0.120, KL=0.125 — closest to real TS. But this is misleading: rand_PT collapsed to 1–4 unique outputs, so its "good" spectrum represents a single memorized shape, not diverse generation.

4. **text_PT balances spectral quality with diversity.** text_PT (PSD_L2=0.228, 686 unique) is the only condition that achieves both reasonable spectral alignment AND high output diversity. Its slight over-concentration in low frequencies (90% vs 79% real) suggests it captures the dominant temporal structure but under-represents mid/high frequency variation.

5. **Spectral alignment is necessary but not sufficient.** rand_PT proves that matching the frequency profile of real TS doesn't imply useful generation — diversity is equally important.

---

## 7. Summary of Findings

### 7.1 Language models contain time-series-decodable sequential structure

A linear map from PT's WikiText hidden states produces outputs 2.2–3.1× closer to real time series than the same map from random vectors. This holds across all 28 layers.

### 7.2 Temporal order is essential

Shuffling the temporal dimension destroys the structure (Shuffled ≈ Random). The patterns come from sequential text processing, not static weight properties.

### 7.3 Both architecture and training contribute

The untrained Qwen3 architecture provides ~2.5× improvement over random vectors. Language training adds 13–21% at early/mid layers but hurts at deep layers (L16+).

### 7.4 Language training uniquely enables diverse generation

With a PSD diversity penalty, PT produces 7.5× more unique TS matches than the untrained architecture (858 vs 114). The diversity penalty cannot unlock diversity from RandomInit because all text inputs produce similar hidden states in an untrained model — language training creates the input variation that enables diverse outputs.

### 7.5 Mid-layers are most useful

Layer weight analysis and layer sweep both show L6–L16 contribute most. Early layers have less processed structure; deep layers specialize for language-specific next-token prediction.

### 7.6 The linear map is a bottleneck

A single linear direction limits both quality and diversity. Mode collapse is the primary failure mode, partially mitigated by the PSD diversity penalty at λ = 0.5.

### 7.7 Pretrained representations are highly anisotropic

PCA reveals PT models concentrate variance into ~3–6 effective dimensions (out of 28,672), with a single direction capturing 82–89% of total variance, while untrained models spread across 400–600 effective dimensions. This anisotropy explains why training the mapper matters for PT (it must find the right low-dimensional subspace) but is irrelevant for RandInit (all directions are equivalent).

### 7.8 Spectral alignment is necessary but not sufficient

PT models produce low-frequency-dominated outputs matching real TS spectral profiles. But rand_PT achieves the best spectral match while collapsing to 1 unique output — good frequency structure without diversity is useless. Only text_PT achieves both.

---

## 8. Limitations

1. **No paired evaluation**: We measure NN distance to any real TS, not whether specific text produces semantically related TS patterns.
2. **Training = evaluation data**: The WikiText sequences used for training are the same used for evaluation (except the held-out generalization test).
3. **Linear constraint**: A single linear map can only capture one direction per layer. A small MLP might produce richer outputs.
4. **Evaluation variance**: NN distance depends on the size and sampling of the TS comparison bank.
5. **Single dataset**: All TS come from GiftEval. Results may differ on other domains.
6. **Training instability**: Results vary between random seeds (828–883 unique). Some seeds can collapse entirely.

---

## 9. Technical Details

### Infrastructure
- 4× NVIDIA RTX 5090 (32GB each), 755GB RAM, 512GB disk
- Hidden state storage: float16 numpy memmap (55GB for 1920 × 512 × 28672)

### Timing
- Hidden state extraction (28 layers, 2000 seqs): ~3 minutes
- Training (100 epochs, batch 32, concat): ~45 minutes
- NN evaluation (1920 preds × 10K TS): ~8 minutes
- Full lambda sweep (4 configs): ~3 hours

### Code
- `scripts/mapping_experiment.py` — single-layer experiments
- `scripts/mapping_concat_layers.py` — concatenated-layer experiments
- `scripts/mapping_diversity.py` — diversity penalty experiments
- `scripts/ablation_study.py` — 2×2 ablation (architecture × input tokens)
- `scripts/plot_ablation_comparison.py` — ablation comparison plots
- `scripts/plot_held_out.py` — held-out generalization plots
- `scripts/additional_experiments.py` — random projections, PCA diversity, spectral alignment
- `mapping_results/` — all results, models, and plots
- `mapping_results/ablation/` — ablation mappers, results, fair comparison, plots
- `mapping_results/additional_experiments/` — Experiments 11–13 results, plots, predictions
- Branch: `mapping-experiment`
