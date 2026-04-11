# Crosscoder Feature Discovery for LLMs and Time-Series Models

## Overview

This project trains a **crosscoder (multi-domain sparse autoencoder)** to discover shared and domain-specific features across:

* A pretrained language model (PT)
* A fine-tuned time-series model (FT)
* A randomly initialized model (RI)

The goal is to:

1. Learn a **shared sparse latent space** across models
2. Identify **interpretable features** (periodicity, pulses, structure, etc.)
3. Compare **geometry vs activation strength** across domains
4. Enable **interpretability via both time-series and text**

---

## Models

We use the following HuggingFace models:

* **FT (time-series trained)**
  `cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512`

* **RI (random baseline)**
  `cerc-aai/Qwen3-0.6B-random-normal_scale-uniform_bin-V512`

* **PT (language model)**
  `Qwen/Qwen3-0.6B`

### Confirmed Architecture (all three models)

* Architecture: `Qwen3ForCausalLM`
* Hidden size: **1024**
* Num layers: **28**
* Max position embeddings: 40960 (base model)
* `tie_word_embeddings: true`

### FT/RI Tokenization

The `cerc-aai` models use **uniform binning with 512 bins (`uniform_bin-V512`)**:
1. Normalize each window to mean 0, std 1 (`normal_scale`)
2. Bin values uniformly over approximately [-5σ, +5σ] into 512 bins
3. Each timestep → **one token ID** (0–511)

> **Note**: The exact context length, prediction length, and tokenizer config must be read directly from the `cerc-aai` model files on load (the HF org hit its private storage quota during initial setup; resolve by freeing storage or downloading weights locally). Look at `training_config.json` and `tokenizer_config.json` inside those repos once accessible.

---

## Core Idea

We train a **shared encoder + per-domain decoders**:

```
x_model → Encoder → z (shared sparse latent)
z → Decoder_model → reconstructed activations
```

Where:

* `z ∈ R^4096`
* Top-K sparsity: **K = 64**
* Input activations: **1024-dimensional hidden states** (residual stream output per transformer layer)

---

## Architecture

### Encoder (2-layer MLP)

```
Input: 1024
→ Linear(1024 → 2048)   [H = 2048]
→ ReLU
→ Linear(2048 → 4096)
→ TopK(k=64)
→ z (sparse)
```

### Decoders (one per domain: PT, FT, RI)

```
z (4096)
→ Linear(4096 → 2048)   [H = 2048]
→ ReLU
→ Linear(2048 → 1024)
→ reconstruction
```

### Important Constraints

* **Top-K sparsity**: exactly 64 active features per example
* **Decoder column normalization**:
  * Each column of the final decoder weight matrix is unit norm
  * Enforced after every gradient step
  * Ensures features are interpretable as directions
* **Separate normalization per domain**:
  * PT, FT, RI each have their own running mean/std (computed over training data)
  * Input activations are z-scored per domain before entering the crosscoder

---

## Data

### Dataset

`Salesforce/GiftEval` on HuggingFace (single `train` split).

* Time series of ~105,120 timesteps at 5-minute frequency
* Multiple univariate series (electricity load data)

### Train / Validation Split

Since GiftEval has only one split, we apply a **temporal split per series**:

* **Training set**: first 70% of each time series (used for crosscoder training)
* **Validation set**: next 15% (used for feature ranking / interpretability analysis)
* **Test set**: final 15% (held out)

Windows are sampled **without overlap** during training, **sliding** during validation.

### Window Size

* **Context length**: infer from `cerc-aai` model `training_config.json`; use **512 timesteps** as default if not found
* Each window is a contiguous segment of `context_length` timesteps

---

## PT Text Representation (Critical)

To compare PT with time-series models, we must align their representations at the **per-timestep** level.

### Step 1: Normalize window

Each time-series window is standardized: `v_norm = (v - mean) / std`

### Step 2: Convert to text

Each timestep value is serialized as a **decimal string rounded to 3 decimal places**, space-separated:

```
"0.123 -0.456 1.034 0.002 ..."
```

The entire window becomes one string. No special separators between timesteps beyond spaces.

### Step 3: Tokenize and run forward pass

* Tokenize the string with the standard Qwen3 tokenizer
* Run through `Qwen/Qwen3-0.6B` to get hidden states at every layer

### Step 4: Align with timesteps

For each timestep `t`:
* Identify which token span in the tokenized sequence corresponds to the text of `v_t` (e.g., `"-0.456"`)
* **Mean-pool the hidden states** across all sub-tokens in that span
* Result: one vector `∈ R^1024` per timestep per layer

This produces **PT activations per timestep ∈ R^1024**, aligned with the FT/RI per-timestep activations.

---

## Training Procedure

### Objective

Minimize reconstruction loss across all domains (equal weighting):

```
L = ||n_PT - D_PT(E(n_PT))||²
  + ||n_FT - D_FT(E(n_FT))||²
  + ||n_RI - D_RI(E(n_RI))||²
  + auxk_coeff * L_auxk
```

Where `E` is the shared encoder (same weights), applied **independently** to each domain's normalized activations. Each domain gets its own sparse latent: `z_PT = E(n_PT)`, `z_FT = E(n_FT)`, `z_RI = E(n_RI)`.

The AuxK loss revives dead features (active in <1% of batches). Dead features are tracked globally across all three domains. For each domain, the top-k dead features by pre-activation are decoded against the reconstruction residual.

### Input to Encoder

At each training step, for a batch of windows, we obtain three activation matrices:
* `x_PT ∈ R^[B×T, 1024]` — PT hidden states (mean-pooled per timestep)
* `x_FT ∈ R^[B×T, 1024]` — FT hidden states
* `x_RI ∈ R^[B×T, 1024]` — RI hidden states

Each domain is normalized independently (z-score with per-domain running EMA stats) and then passed through the shared encoder independently, producing per-domain sparse latents `z_PT`, `z_FT`, `z_RI`.

### Training Steps

1. Sample batch of windows from training portion of `Salesforce/GiftEval`
2. Compute activations per domain (FT, RI: direct hidden states; PT: text pipeline + mean pooling)
3. Normalize per domain (z-score with running EMA stats)
4. Encode each domain independently through the shared encoder → `z_PT`, `z_FT`, `z_RI`
5. Decode each: `D_PT(z_PT)`, `D_FT(z_FT)`, `D_RI(z_RI)`
6. Compute reconstruction loss (sum of per-domain MSEs) + AuxK loss for dead features
7. Backprop + Adam step
8. Normalize decoder columns (unit norm per column of final linear layer)

### Hyperparameters (proposed defaults)

| Parameter | Value |
|-----------|-------|
| Latent dim | 4096 |
| K (top-k) | 64 |
| H (hidden dim) | 2048 |
| Batch size (windows) | 64 |
| Effective batch (timesteps) | 64 × context_length |
| Learning rate | 3e-4 |
| LR schedule | Cosine with warmup (1000 steps) |
| Training steps | 100,000 |
| Optimizer | AdamW (β1=0.9, β2=0.999, wd=0.0) |
| Dead neuron threshold | feature active < 1% of batches over last 1000 steps |
| Dtype | bfloat16 (models), float32 (crosscoder) |

### Multi-GPU Strategy (4× RTX 5090, 32GB each)

One crosscoder is trained **per layer** (28 total). Since each crosscoder is small (~40M params), and the bottleneck is activation extraction from the three ~0.6B models:

* **Activation extraction**: All 3 models are loaded on each GPU (3 × ~1.2GB = 3.6GB in bfloat16), with data parallelism across the 4 GPUs.
* **Crosscoder training**: 28 crosscoders are assigned across 4 GPUs (7 per GPU), trained in parallel. Each GPU holds its 7 crosscoders and a copy of all 3 models.
* **Flow per step**: sample windows → extract activations for all 28 layers in one forward pass per model → route each layer's activations to the appropriate crosscoder on the assigned GPU.

---

## Feature Extraction

After training, for each layer:

### Step 1: Collect activations

Run the **validation portion** of GiftEval and store:
* Latent activations `z` per timestep
* Corresponding raw windows
* Reconstructed outputs per domain

### Step 2: Rank windows per feature

For each feature `f` (0–4095):
1. Collect all `z_f` values across the validation set
2. Select top-100 activating timesteps
3. Retrieve the containing window and the timestep position within it

This gives: **top-100 activating windows per feature**

### Step 3: Compute statistics

For each feature:
* Mean activation per domain (PT, FT, RI)
* Activation frequency (fraction of timesteps where feature is active)
* Reconstruction contribution per domain
* Cosine similarity between domain decoder vectors for this feature

---

## Feature Categorization

We classify features into:

* **PT_FT**: strong in both PT and FT, weak in RI
* **FT_RI**: strong in FT and RI, weak in PT
* **PT_RI**: strong in PT and RI, weak in FT
* **Universal**: strong across all three domains
* **PT_only / FT_only / RI_only**: strong in one domain only
* **Mixed**: no clear pattern

### Heuristics

A domain is "strong" for a feature if its mean activation is above the median across all features and domains.

Also use:
* Cosine similarity of per-domain decoder vectors for the feature
* Reconstruction loss contribution per domain

---

## Visualization & Analysis Output

For each feature, generate:

1. **Top windows plot**: grid of top-10 activating windows, with activation magnitude shown per timestep (as a color bar or overlay)
2. **Activation histogram**: distribution of `z_f` values
3. **Domain comparison bar chart**: mean activation for PT, FT, RI
4. **Decoder vector cosine similarity matrix**: PT↔FT, FT↔RI, PT↔RI

### PT Text Display

For each top-activating window, also display:
* The raw text string fed to PT (decimal values)
* Highlighted tokens where activation is highest (for qualitative analysis)

### Saved Data Format

Each feature's analysis is saved as a JSON file for easy manual review and scoring:

```json
{
  "layer": 8,
  "feature_id": 42,
  "category": "PT_FT",
  "activation_stats": {
    "PT": {"mean": 0.8, "std": 0.2, "freq": 0.15},
    "FT": {"mean": 1.2, "std": 0.3, "freq": 0.20},
    "RI": {"mean": 0.1, "std": 0.05, "freq": 0.02}
  },
  "decoder_cosine_sim": {"PT_FT": 0.91, "FT_RI": 0.12, "PT_RI": 0.10},
  "top_windows": [
    {
      "window_id": "series_7_offset_1024",
      "activation_at_timestep": 312,
      "activation_value": 3.41,
      "raw_values": [0.12, -0.45, ...],
      "pt_text": "0.120 -0.450 ...",
      "reconstruction_loss_PT": 0.02,
      "reconstruction_loss_FT": 0.01,
      "reconstruction_loss_RI": 0.18
    },
    ...
  ],
  "interpretability_score": null,
  "notes": ""
}
```

The `interpretability_score` (1–10) and `notes` fields are filled in manually during analysis.

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| 9–10 | Clean, obvious pattern (e.g., pure sine wave, sharp spike train) |
| 7–8 | Mostly clear structure with some noise |
| 5–6 | Partial structure visible |
| 3–4 | Weak or ambiguous pattern |
| 1–2 | No discernible structure |

---

## Output Structure

```
checkpoints/
  layer_{i}/
    crosscoder.pt         ← trained weights
    norm_stats.json       ← per-domain mean/std

analysis/
  layer_{i}/
    features.json         ← summary stats for all 4096 features
    feature_{j}.json      ← per-feature detail (top windows, stats)
    plots/
      feature_{j}_windows.png
      feature_{j}_hist.png
      feature_{j}_domains.png
    catalog.csv           ← all features with score + notes columns
```

---

## Code Structure

```
src/
  config.py               ← dataclass for all hyperparameters
  data/
    dataset.py            ← GiftEval loader, windowing, train/val split
    tokenize.py           ← uniform-bin tokenizer (FT/RI) + text serializer (PT)
  models/
    extractor.py          ← load all 3 models, extract hidden states for all layers
  crosscoder/
    model.py              ← Encoder, Decoder, Crosscoder nn.Module
    train.py              ← training loop, dead-neuron monitoring, norm enforcement
  analysis/
    feature_extract.py    ← run validation set, collect z activations
    ranking.py            ← top-N windows per feature
    categorize.py         ← PT_FT / FT_RI / Universal etc.
  visualization/
    plots.py              ← all matplotlib plotting functions

scripts/
  train_all_layers.py     ← main entry point: trains all 28 crosscoders across 4 GPUs
  extract_features.py     ← runs validation set, saves per-feature JSONs
  analyze_features.py     ← generates plots + catalog.csv
  score_features.py       ← interactive CLI to assign 1–10 scores to features
```

---

## Key Insights to Look For

1. **Periodicity features** — do PT and FT both activate on sine-like patterns?
2. **Pulse / spike patterns** — sharp local events
3. **Shared geometry vs magnitude differences** — same direction, different scale across domains
4. **Increasing universality with depth** — do deeper layers have more Universal features?
5. **FT amplification over PT/RI** — FT should have stronger signal structure

---

## Reproducibility Checklist

* [ ] Load all 3 models (resolve cerc-aai HF storage issue)
* [ ] Infer context length and prediction length from `cerc-aai` model `training_config.json`
* [ ] Implement PT text conversion (3 d.p. decimal strings) + mean pooling
* [ ] Implement uniform-bin tokenizer for FT/RI (infer exact bins from model tokenizer)
* [ ] Normalize per domain
* [ ] Train 28 crosscoders (one per layer) with:
  * latent_dim = 4096, K = 64, H = 2048
* [ ] Use temporal-split training portion of `Salesforce/GiftEval` for training
* [ ] Use temporal-split validation portion for feature ranking
* [ ] Extract top-100 activating windows per feature
* [ ] Generate plots + JSON outputs
* [ ] Perform manual interpretability scoring (1–10)

---

## Notes

* This is **not compression** — it is **feature discovery**
* Latent space is **overcomplete** (4096 features for 1024-dim input)
* Sparsity is essential for interpretability
* PT provides **semantic grounding via text**
* FT provides **strong signal structure**
* The `cerc-aai` models use the **same Qwen3 transformer architecture** as the base model; only the tokenizer and weights differ

---

## Future Extensions

* Vary K (e.g., 32, 128)
* Add more domains
* Replace Top-K with learned sparsity
* Cluster features automatically
* Quantify feature overlap across layers
