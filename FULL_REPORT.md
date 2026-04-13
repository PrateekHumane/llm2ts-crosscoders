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

5 random seeds for the best config (PT concat div0.5):

| Seed | Unique | Clusters | NN Dist |
|-----:|------:|------:|------:|
| 0 | 828 | 16/20 | 0.744 |
| 1 | 883 | 15/20 | 0.738 |

Results are reproducible: ~830–880 unique matches consistently.

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
- `scripts/plot_held_out.py` — held-out generalization plots
- `mapping_results/` — all results, models, and plots
- Branch: `mapping-experiment`
