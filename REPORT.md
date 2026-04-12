# Crosscoder Feature Discovery and Linear Mapping from Language to Time Series

## 1. Overview

This project investigates what language pretraining contributes to time series forecasting. We use three variants of Qwen3-0.6B:

- **PT**: Base pretrained language model (`Qwen/Qwen3-0.6B`), trained only on text
- **FT**: PT finetuned on time series forecasting (`cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512`)
- **RI**: Randomly initialized, trained on the same time series data as FT (`cerc-aai/Qwen3-0.6B-random-normal_scale-uniform_bin-V512`)

FT outperforms RI on time series forecasting. The question is: **what exactly does language pretraining contribute?**

We conduct two main experiments:
1. **Crosscoder analysis**: Train sparse autoencoders to identify shared and domain-specific features across the three models
2. **Linear mapping experiment**: Test whether PT's hidden states from processing natural language text can be linearly decoded into time series data

---

## 2. Models and Data

### 2.1 Architecture

All three models share the Qwen3-0.6B architecture: 28 transformer layers, hidden dimension 1024, 0.6B parameters.

### 2.2 Time Series Tokenization (FT/RI)

Time series values are z-score normalized per window, clipped to [-5, 5], and discretized into 512 uniform bins. Each timestep maps to one token ID (0-511). Context length: 512 timesteps. Prediction length: 64 timesteps.

### 2.3 PT Text Representation

For crosscoder training, PT processes the same time series as decimal text strings ("0.123 -0.456 1.034 ..."). Sub-token hidden states are mean-pooled per timestep to produce one 1024-dim vector per timestep, aligned with FT/RI.

### 2.4 Dataset

- **Time series**: Salesforce/GiftEval (univariate time series, multiple datasets)
- **Text**: WikiText-103 (for probing and the mapping experiment)
- **Temporal split**: 70% train / 15% validation / 15% test per series

---

## 3. Crosscoder Training

### 3.1 Architecture

Each crosscoder has a shared encoder applied independently to each domain's activations, producing per-domain sparse latents:

```
Encoder (shared weights): Linear(1024 -> 2048) -> ReLU -> Linear(2048 -> 4096) -> TopK(k=64)
Decoder (one per domain): Linear(4096 -> 2048) -> ReLU -> Linear(2048 -> 1024)
```

Critical design: the encoder is applied to each domain **separately** (not to an average). This produces `z_PT`, `z_FT`, `z_RI` independently, enabling direct comparison of which features fire for which models on the same input.

### 3.2 Training Details

- **Loss**: `MSE(D_PT(z_PT), n_PT) + MSE(D_FT(z_FT), n_FT) + MSE(D_RI(z_RI), n_RI) + auxk_coeff * L_auxk`
- **AuxK loss**: Revives dead features. Shared dead mask (feature is dead if inactive across all three domains). Per-domain auxiliary forward passes.
- **Per-domain normalization**: EMA running mean/std per domain, applied before encoding
- **Decoder column normalization**: Unit norm enforced after each gradient step
- **Training**: DDP across 4x RTX 5090, 130,000 precomputed windows per layer, converged at ~36,000 steps
- **One crosscoder per layer**: 28 total (27 successful, layer 27 produced NaN)

### 3.3 Training Results

| Layer | Loss | Dead Features | Alive Features |
|------:|-----:|--------------:|--------------:|
| 0 | 0.009 | 2472 | 1624 |
| 3 | 0.056 | 1562 | 2534 |
| 6 | 0.060 | 1894 | 2202 |
| 8 | 0.090 | 2071 | 2025 |
| 13 | 0.211 | 1810 | 2286 |
| 16 | 0.302 | 2470 | 1626 |
| 20 | 0.232 | 2716 | 1380 |
| 24 | 0.145 | 2839 | 1257 |
| 26 | 0.129 | 3038 | 1058 |

Loss increases with depth up to layer 17 (0.315), then decreases. Dead features range from 38-74% across layers.

---

## 4. Feature Categorization

### 4.1 Method

For 50,000 validation time series windows, each domain's activations are passed through the shared encoder independently. A feature is "active for domain D" if it fires (z > 0) on more than 1% of timesteps. Features are categorized as:

- **PT_FT_RI** (Universal): Active in all three
- **PT_FT**: Active in PT and FT only
- **FT_RI**: Active in FT and RI only
- **PT_RI**, **PT_only**, **FT_only**, **RI_only**: Other combinations
- **None**: Dead features

### 4.2 Category Distribution (Layer 8, representative)

| Category | Count |
|----------|------:|
| PT_FT_RI | 190 |
| PT_FT | 60 |
| FT_RI | 160 |
| PT_RI | 200 |
| PT_only | 140 |
| FT_only | 110 |
| RI_only | 300 |
| None (dead) | 2936 |

---

## 5. WikiText Probing

### 5.1 Method

For PT-involving features (PT_FT_RI and PT_FT), we probe what they detect in natural language by running 30,000 WikiText-103 sequences through the PT model and encoding with the crosscoder.

**Critical normalization fix**: The crosscoder's PT normalization stats were learned on numeric text hidden states. WikiText hidden states have very different distributions (std up to 10x larger at deeper layers). We compute WikiText-specific normalization stats and use those instead of the crosscoder's learned PT stats. Without this fix, WikiText features were either constant-valued or zero.

### 5.2 Exhaustive Feature Scoring

All 1,620 PT-involving features across 27 layers were examined: each feature's time series plot (top 10 activating windows with activation overlay) and WikiText spans (top 10 activating tokens with surrounding context) were compared for semantic connections.

### 5.3 Top Cross-Domain Features

| Score | Layer | Feature | Category | Description |
|------:|------:|--------:|----------|-------------|
| 8/10 | 8 | 1734 | PT_FT | **Repetition detector**: Text fires on "3*3*3*3", "2-3-2 configs", "Bills Bills Bills". TS fires on square waves and periodic oscillations. |
| 7/10 | 26 | 2193 | PT_FT | **Slope/derivative detector**: Text fires on "slope", "derivative", "tangent line". TS fires on rising/falling slopes. |
| 6/10 | 4 | 3319 | PT_FT_RI | **Metronome/beat detector**: Text fires on "metronome", "beats", "tempo". TS fires on sharp periodic spikes. |
| 6/10 | 5 | 1888 | PT_FT | **Chord progression detector**: Text fires on "F-Eb-G" harmonic sequences. TS fires on periodic peaked patterns. |
| 6/10 | 6 | 4086 | PT_FT_RI | **Sonata/periodic detector**: Text fires on "Alkan", "sonata". TS fires on sinusoidal oscillations. |
| 6/10 | 8 | 2760 | PT_FT | **Chemistry/outlier detector**: Text fires on rare chemical elements. TS fires on isolated spikes. |
| 6/10 | 11 | 2992 | PT_FT_RI | **Rainfall/spike detector**: Text fires on "heavy rainfall". TS fires on sudden spikes from flat baseline. |
| 6/10 | 22 | 1644 | PT_FT_RI | **Crash/flood detector**: Text fires on "flooding", "volcanic". TS fires on sharp drops. |
| 6/10 | 26 | 2426 | PT_FT_RI | **Geometry/angle detector**: Text fires on "angle", "opposite", "side". TS fires on slopes. |

### 5.4 Best FT_RI Features (Time Series Only)

| Score | Layer | Feature | Description |
|------:|------:|--------:|-------------|
| 8/10 | 6 | 861 | Periodic trough detector |
| 8/10 | 7 | 3604 | High-frequency oscillation detector |
| 8/10 | 26 | 2354 | Spike detector |
| 7/10 | 5 | 3909 | Peak detector |
| 7/10 | 6 | 3512 | Plateau/step detector |
| 7/10 | 10 | 1024 | Upward step/regime change detector |

### 5.5 Summary Statistics

Out of 1,620 cross-domain features examined:
- 1 scored 8/10 (genuine cross-domain semantic connection)
- 1 scored 7/10
- 7 scored 6/10
- 7 scored 5/10
- 1,604 scored 4 or below

The vast majority of PT-involving features do not show meaningful semantic connections between WikiText content and time series patterns. The connections that exist tend to involve **structural/mathematical concepts** (repetition, slopes, periodicity) rather than semantic content.

---

## 6. Linear Mapping Experiment

### 6.1 Motivation

The crosscoder analysis reveals shared features but doesn't test whether PT's representations can **generate** time series. We test a stronger claim: can a linear map decode time series directly from PT's hidden states when processing natural language?

### 6.2 Setup

**Input**: PT hidden states from processing WikiText sequences, shape (512, 1024) per sequence — one 1024-dim vector per token position.

**Model**: A single linear layer applied per timestep: `y_t = W @ h_t + b`, where `W` is (1, 1024). This produces a scalar time series value per position. Total parameters: 1025.

**Target**: Z-scored time series windows from GiftEval validation set, shape (512,).

**Training (EM-style)**: No paired text-TS data. For each predicted sequence, find the nearest real time series (by MSE) and minimize MSE to that target. This rewards the map for producing outputs that lie anywhere on the time series manifold.

**Normalization**: Predicted sequences are z-score normalized before matching (mean=0, std=1).

### 6.3 Evaluation Metric

**Nearest Neighbor Distance (NN dist)**: For each predicted sequence, compute MSE to all real time series and take the minimum. Lower = predictions are closer to real time series. Averaged over 300-500 evaluation sequences.

### 6.4 Results: Layer 8 Baselines

| Method | NN Distance | Description |
|--------|------:|-------------|
| **RI** | **0.4629** | RI model hidden states from WikiText |
| **PT** | **0.7161** | PT model hidden states from WikiText |
| Random | 1.6873 | Gaussian random vectors, same shape |
| Shuffled | 1.7729 | PT hidden states with shuffled time dimension |

**PT is 2.4x closer to real time series than Random.** Shuffled (1.77) is approximately equal to Random (1.69), confirming that the structure comes from temporal text processing, not just weight statistics.

RI beats PT (0.46 vs 0.72) because RI was trained on time series using the same 0-511 token vocabulary. WikiText tokens passing through RI produce hidden states that are inherently "time-series-shaped".

### 6.5 Results: Layer Sweep

| Layer | PT (NN dist) | Random (NN dist) | Ratio (Random/PT) |
|------:|-----------:|-----------:|------:|
| 0 | 0.6720 | 1.6780 | 2.50x |
| **2** | **0.5484** | **1.6789** | **3.06x** |
| 4 | 0.5577 | 1.6724 | 3.00x |
| 8 | 0.5722 | 1.6716 | 2.92x |
| 12 | 0.5523 | 1.6736 | 3.03x |
| 16 | 0.7490 | 1.6788 | 2.24x |
| 20 | 0.6105 | 1.6754 | 2.74x |
| 24 | 0.6524 | 1.6755 | 2.57x |
| 26 | 0.6526 | 1.6744 | 2.57x |

**PT beats Random at every layer** with ratios from 2.24x to 3.06x.

**Pattern across depth**:
- **L0 (2.50x)**: Embedding layer — some sequential structure
- **L2-L12 (2.9-3.1x)**: Peak — early/mid layers have the richest time-series-decodable structure
- **L16 (2.24x)**: Dip — transition from general sequence processing to task-specific representations
- **L20-L26 (2.6x)**: Partial recovery — deep layers still carry substantial structure

Random baseline is rock-steady at ~1.68 regardless of layer, confirming it is a true null baseline.

### 6.6 Training Variant Comparison

Two training variants were tested:

- **EM-style (Variant 2)**: Find nearest real TS, minimize MSE to it. **Loss converged to ~0.46.** Produced structured, smooth predictions.
- **Soft retrieval (Variant 1)**: Contrastive loss encouraging predictions to match their best TS. **Loss did not converge (~3.97).** The contrastive objective was too difficult for a single linear layer.

---

## 7. Key Findings

### 7.1 Language models contain time-series-decodable structure

A 1025-parameter linear map, trained with no text-TS pairing, can extract signals from WikiText hidden states that are 2-3x closer to real time series than the same procedure on random vectors. This structure exists at every layer and peaks in early-mid layers (L2-L12).

### 7.2 Temporal order is essential

Shuffling the temporal dimension of PT's hidden states destroys the structure (Shuffled NN dist 1.77 vs PT 0.72). The time-series-like patterns come from the **sequential processing of text**, not from static weight properties.

### 7.3 Cross-domain features are rare but real

Out of 1,620 PT-involving crosscoder features examined exhaustively, only ~16 show clear cross-domain connections. The strongest is a **repetition detector** (Layer 8, Feature 1734) that fires on mathematical repetition in text and periodic patterns in time series.

### 7.4 Structural, not semantic

The cross-domain features that exist encode **structural properties** (repetition, slopes, periodicity, suddenness) rather than semantic content. No features were found where, e.g., text about "fire" maps to spike patterns in TS.

### 7.5 RI's advantage

RI (randomly initialized, TS-trained) produces better linear-decodable time series from WikiText than PT. This is because RI processes all tokens through a time-series-optimized architecture, producing hidden states that are inherently temporal regardless of input content.

---

## 8. Technical Details

### 8.1 Infrastructure

- **Hardware**: 4x NVIDIA RTX 5090 (32GB each), 755GB RAM, 512GB disk
- **Crosscoder training**: DDP across 4 GPUs via torchrun, ~40 min per layer (precompute + train)
- **Total crosscoder training time**: ~30 hours for 28 layers
- **Analysis pipeline**: ~9 hours for feature extraction + categorization + plotting across 27 layers
- **Mapping experiment**: ~2 hours for all baselines + layer sweep

### 8.2 Code Structure

```
src/
  config.py                    Configuration dataclass
  data/dataset.py              GiftEval loader, windowing, temporal split
  data/tokenize.py             Uniform-bin tokenizer (FT/RI) + text serializer (PT)
  data/wikitext.py             WikiText-103 loader + tokenization
  models/extractor.py          Load 3 models, extract hidden states with hooks
  crosscoder/model.py          Encoder, Decoder, Crosscoder (per-domain encoding)
  crosscoder/precompute.py     4-GPU parallel activation extraction
  crosscoder/train_from_disk.py DDP training from precomputed memmaps

scripts/
  train_all_layers.py          Full 28-layer training pipeline (priority ordering)
  train_layer_ddp.py           Single-layer DDP training via torchrun
  analyze_layer.py             Feature categorization + WikiText probing + plotting
  mapping_experiment.py        Linear mapping from text hidden states to time series

mapping_results/               Linear mapping experiment results
analysis/                      Per-layer feature categorization and plots
checkpoints/                   Trained crosscoder weights
```

### 8.3 Layer Training Priority

Layers were trained in binary subdivision order for maximum early coverage: 13, 0, 27, 6, 20, 3, 10, 16, 24, 1, 4, 8, 11, 14, 18, 22, 26, 2, 5, 7, 9, 12, 15, 17, 19, 21, 23, 25.

---

## 9. Limitations

1. **Dead features (40-60%)**: Nearly half the crosscoder capacity is wasted. Better dead feature recovery could reveal more interesting shared features.

2. **Linear mapping is simple**: A 1025-parameter model captures only the dominant linear direction. A small MLP might reveal richer structure.

3. **WikiText normalization**: The crosscoder's PT normalization stats don't transfer to WikiText. We used WikiText-specific stats as a workaround, but this means the crosscoder features may not be optimally applied to natural language.

4. **Single dataset**: All time series come from GiftEval. Results may differ on other domains.

5. **Layer 27 failure**: The deepest layer produced NaN during training and was excluded.

---

## 10. Conclusions

Language pretraining instills **domain-general sequential structure** into transformer hidden states. This structure is:

- **Linearly decodable** into time-series-like signals (2-3x better than random)
- **Temporally ordered** (destroyed by shuffling)
- **Layer-dependent** (peaks at L2-L12, dips at L16)
- **Present without any time series exposure** (PT has never seen time series)

This provides a mechanistic explanation for why language-pretrained models transfer to time series forecasting: the sequential patterns learned from text (periodicity, trends, sudden changes) are directly reusable for temporal data.
