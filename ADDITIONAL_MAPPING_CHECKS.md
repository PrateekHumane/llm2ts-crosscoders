# Additional Experiments for NeurIPS Submission

This document specifies three critical experiments to strengthen the causal and mechanistic claims of the time-series decoding project.

All experiments should reuse existing infrastructure where possible (hidden state extraction, evaluation pipeline, etc.).

---

# Experiment 1: Train-Free Random Projection Baseline

## Goal

Determine whether alignment between LLM hidden states and time series is intrinsic to representation geometry, not learned by the linear mapper.

---

## Setup

For each of the 4 ablation conditions:

- text_PT
- text_RandInit
- rand_PT
- rand_RandInit

Use the same hidden states already computed (concatenated layers, D = 28672).

---

## Procedure

1. Remove training entirely
   - Do NOT train a linear mapper
   - Do NOT run EM matching

2. Sample random linear projections
   - Sample W ~ N(0, I)
   - Shape: (1, D)
   - Bias b = 0

3. For each W:
   - Compute predictions:
     y_hat[i, t] = W @ h[i, t]
   - Apply same normalization:
     y_hat[i] = (y_hat[i] - mean(y_hat[i])) / max(std(y_hat[i]), 1e-4)

4. Evaluate using existing metrics:
   - NN distance (raw + deduped)
   - Unique matches
   - Cluster coverage
   - Entropy

5. Repeat for multiple random projections:
   - At least 20–50 different W samples
   - Report mean and std

---

## Output

For each condition:

Condition | NN Dist | Unique | Clusters | Entropy

Also report variance across random projections.

---

## Expected Insight

- If text_PT > text_RandInit without training → intrinsic structure
- If rand_PT collapses → confirms input dependence
- Establishes that results are not purely due to learned mapping

---

# Experiment 2: Hidden State Diversity Analysis

## Goal

Measure whether diversity differences arise from representation geometry (rank / variance / spread).

---

## Setup

Use hidden states H for all 4 ablation conditions.

Concatenated layers (D = 28672).

---

## Procedure

1. Flatten hidden states

For each condition:

- Collect all sequences:
  H has shape (N, T, D)

- Reshape into:
  H_flat has shape (N*T, D)

---

2. Compute statistics

(A) Per-dimension variance

- Compute variance across samples:
  var_d = variance(H_flat[:, d])

- Report:
  - mean variance
  - optionally histogram of variances

---

(B) PCA spectrum

- Run PCA on H_flat (or a large random subset if memory constrained)
- Compute eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_D

Report:
- Top 100 eigenvalues
- Plot spectrum (log scale)

---

(C) Effective rank

Compute:

p_i = λ_i / sum(λ)
H_entropy = -sum(p_i * log(p_i))
rank_eff = exp(H_entropy)

---

(D) Participation ratio (optional)

PR = (sum(λ))^2 / sum(λ^2)

---

## Output

Condition | Mean Var | Effective Rank | Participation Ratio

Also include:
- PCA spectrum plots (all 4 overlaid)

---

## Expected Insight

- rand_PT → very low rank (collapse)
- text_PT → high rank (diverse manifold)
- rand_RandInit > text_RandInit → input entropy effect

This directly supports:
diversity = representation spread

---

# Experiment 3: Spectral Alignment Analysis

## Goal

Test whether LLM outputs match time series due to frequency structure alignment.

---

## Setup

Use generated sequences from trained mappers:

- text_PT
- text_RandInit
- rand_PT
- rand_RandInit

Also include:
- real time series from GiftEval

Use held-out predictions if available.

---

## Procedure

1. Compute Power Spectral Density (PSD)

For each sequence y:

PSD(y) = |FFT(y)|^2

Normalize PSD:

PSD = PSD / sum(PSD)

---

2. Aggregate distributions

For each condition:

- Collect PSDs across all sequences
- Compute:
  - mean PSD
  - variance of PSD

---

3. Compare to real TS

Compute distances:

(A) L2 distance:
|| PSD_model - PSD_real ||^2

(B) KL divergence:
KL(PSD_model || PSD_real)

(C) Optional: Wasserstein distance

---

4. Frequency band analysis

Split frequencies into:

- Low (0–10%)
- Mid (10–50%)
- High (50–100%)

Compute energy in each band.

---

## Output

Condition | PSD L2 | KL | Low Freq | Mid Freq | High Freq

Also include:
- Mean PSD curves (all conditions + real TS)
- Frequency band bar chart

---

## Expected Insight

- Real TS → low-frequency dominated
- Random → flat spectrum
- RandomInit → low-frequency bias
- text_PT → closest match to real TS

Supports hypothesis:
Transformers succeed due to spectral alignment

---

# Notes

- Use held-out text where possible
- Keep evaluation consistent with existing pipeline
- Save all intermediate outputs for plotting
- Prefer batched FFT for speed

---

# Deliverables

Each experiment should produce:

1. Tables (CSV + markdown)
2. Plots (PNG)
3. Short summary (2–3 bullet points per experiment)