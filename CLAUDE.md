# CLAUDE.md — Project Context for llm2ts-crosscoders

## What This Project Does

Trains a crosscoder (multi-domain sparse autoencoder) to discover shared and domain-specific features across three variants of Qwen3-0.6B:
- **PT**: `Qwen/Qwen3-0.6B` — standard pretrained language model
- **FT**: `cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512` — fine-tuned on time series
- **RI**: `cerc-aai/Qwen3-0.6B-random-normal_scale-uniform_bin-V512` — randomly initialized baseline

One crosscoder per transformer layer (28 total). Each crosscoder: shared encoder applied independently per domain → per-domain sparse latents (4096-dim, Top-K=64) → 3 per-domain decoders. The encoder weights are shared but each domain is encoded separately.

**READ THE README.md BEFORE CODING. It contains the full architecture, hyperparameters, and design decisions.**

## Hardware

4× NVIDIA RTX 5090 (32GB each). Use all 4 GPUs whenever training or doing large-scale inference.

## Key Technical Decisions (already agreed with user)

- H (MLP hidden dim in encoder/decoder) = **2048**
- Top-K = **64**, latent dim = **4096**
- PT text format: each timestep as **decimal string rounded to 3 decimal places**, space-separated
- Equal loss weighting across PT, FT, RI
- Dataset: `Salesforce/GiftEval` (HuggingFace). Temporal split: 70% train / 15% val / 15% test
- Context length: **infer from `cerc-aai` model's `training_config.json`**; default to 512 if unavailable
- One crosscoder per layer (28 crosscoders total), trained in parallel across 4 GPUs

## HuggingFace Access

HF token is in `~/.bashrc` as `HF_TOKEN`. Source it before running anything:
```bash
source ~/.bashrc
```

The `cerc-aai` models were inaccessible during initial setup due to **HF private storage quota being exceeded**. Once resolved:
1. Load them to confirm `context_length` from `training_config.json`
2. Inspect tokenizer to confirm exact binning range and special tokens
3. Check `config.json` for any architecture differences from base `Qwen/Qwen3-0.6B`

## FT/RI Tokenization (uniform_bin-V512)

The name tells us: normalize values → uniform binning → 512 bins. Implementation:
1. Compute mean and std of the context window
2. Standardize: `v_norm = (v - mean) / std`
3. Clip to [-5, 5]
4. Map uniformly to bins 0–511: `bin = int((v_norm + 5) / 10 * 512)`
5. **Verify this against the actual tokenizer once models are accessible**

## PT Text Representation

```python
text = " ".join(f"{v:.3f}" for v in normalized_window)
```
Then tokenize with `Qwen/Qwen3-0.6B` tokenizer. Mean-pool hidden states of sub-tokens that correspond to each timestep's value string to produce one 1024-dim vector per timestep.

## Code Structure (not yet written)

```
src/
  config.py
  data/dataset.py, tokenize.py
  models/extractor.py
  crosscoder/model.py, train.py
  analysis/feature_extract.py, ranking.py, categorize.py
  visualization/plots.py
scripts/
  train_all_layers.py
  extract_features.py
  analyze_features.py
  score_features.py
```

## Analysis Output

Per-feature JSON files with top-100 activating windows, activation stats per domain, decoder cosine similarities. A `catalog.csv` with columns: Layer, Feature ID, Category, Score (1–10), Description, Notes.

The `score_features.py` script should provide an interactive CLI to view plots + text and assign 1–10 interpretability scores.

## Monitoring During Training

Watch for:
- Dead features (active < 1% of batches over last 1000 steps) — log count
- Training loss divergence
- Decoder column norms drifting from 1.0

## Style

- Python + PyTorch
- Scripts as entry points (not a pip-installable package)
- bfloat16 for models, float32 for crosscoder parameters
- Configuration via dataclass in `src/config.py` (not argparse soup)
- No unnecessary abstractions
