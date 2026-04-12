# Experiment: Decoding Time Series from PT Hidden State Sequences via Linear Mapping and Retrieval Matching

## 1. Objective

Test whether **hidden state sequences from a pretrained language model (PT)**, when processing natural language (WikiText), contain sufficient structure to generate realistic time series signals via a **simple linear sequence-to-sequence mapping**.

We specifically test whether:
- A **linear map from hidden state sequences → time series sequences** can produce outputs that lie close to real time series data.
- This can be achieved **without explicit supervision or aligned pairing**, using only **retrieval-based matching**.

---

## 2. Setup

### 2.1 Models
- PT: pretrained language model (frozen)
- FT: finetuned model (used only for evaluation / optional controls)
- RI: randomly initialized model (baseline)

---

### 2.2 Data

#### Text
- Dataset: WikiText
- Input length: 512 tokens

#### Time Series
- Dataset: GiftEval
- Window length: 512 timesteps
- Values:
  - Either continuous (z-scored)
  - Or discretized (same as FT training)

---

## 3. Representations

### 3.1 Hidden State Sequences

For each text sequence:
- Pass through PT
- Extract hidden states from layer `L`:

\[
H_i \in \mathbb{R}^{T \times d}
\]

Where:
- \( T = 512 \)
- \( d = 1024 \)

---

### 3.2 Time Series

Each time series window:

\[
Y_j \in \mathbb{R}^{T}
\]

---

## 4. Model: Linear Sequence-to-Sequence Mapping

We learn a **single linear layer applied per timestep**:

\[
\hat{y}_{i,t} = W h_{i,t} + b
\]

Where:
- \( W \in \mathbb{R}^{1 \times d} \)
- Output:
\[
\hat{Y}_i \in \mathbb{R}^{T}
\]

This is equivalent to:
- Applying the same linear probe at every timestep

---

## 5. Similarity Function (Multi-Metric)

Define similarity between predicted and real time series:

\[
s(\hat{Y}, Y) = - \lambda_1 \cdot \text{MSE} - \lambda_2 \cdot \text{PSD\_dist} - \lambda_3 \cdot \text{ACF\_dist}
\]

Where:

### 5.1 MSE
\[
\text{MSE} = \frac{1}{T} \sum_t (\hat{y}_t - y_t)^2
\]

### 5.2 PSD Distance
- Compute FFT magnitude
- Compare L2 distance

### 5.3 Autocorrelation Distance
- Compute autocorrelation function
- Compare L2 distance

---

## 6. Training Objectives

We test two variants.

---

### 6.1 Variant 1: Soft Retrieval (No Fixed Pairing)

For each predicted sequence \( \hat{Y}_i \):

- Compare against batch of real time series \( \{Y_j\}_{j=1}^B \)

Compute similarities:
\[
s_{ij} = s(\hat{Y}_i, Y_j)
\]

Define loss:
\[
\mathcal{L}_i = -\log \frac{\exp(\max_j s_{ij})}{\sum_j \exp(s_{ij})}
\]

Interpretation:
- Model is rewarded if prediction matches **any real time series**
- No explicit pairing required

---

### 6.2 Variant 2: EM-Style Hard Matching

Repeat:

#### E-step:
Assign best match:
\[
j^* = \arg\max_j s(\hat{Y}_i, Y_j)
\]

#### M-step:
Optimize:
\[
\mathcal{L}_i = \| \hat{Y}_i - Y_{j^*} \|^2
\]

Interpretation:
- Model learns to map each text sequence to its closest time series
- Equivalent to clustering in signal space conditioned on hidden states

---

## 7. Training Details

- Batch size: B (e.g., 64–256)
- Optimizer: Adam
- Learning rate: 1e-3 (tune)
- Train until convergence

---

## 8. Evaluation

### 8.1 Nearest Neighbor Retrieval

For each predicted \( \hat{Y}_i \):

\[
j^* = \arg\min_j \| \hat{Y}_i - Y_j \|
\]

Metrics:
- Mean nearest-neighbor distance
- Top-k retrieval accuracy

---

### 8.2 Distributional Similarity

Compare generated vs real:

- PSD distribution
- Autocorrelation distribution
- Variance / smoothness statistics

---

### 8.3 Structural Metrics

Check whether predictions exhibit:
- Periodicity
- Trends
- Smoothness

Compare against:
- Real data
- Random noise baseline

---

## 9. Baselines

### 9.1 Random Hidden States
- Replace \( H_i \sim \mathcal{N}(0, I) \)

---

### 9.2 RI Model
- Use hidden states from RI instead of PT

---

### 9.3 Shuffled Time Dimension
- Shuffle timesteps in \( H_i \)

---

### 9.4 Shuffled Dataset
- Shuffle time series dataset

---

## 10. Ablations

### 10.1 Layer Sweep
- Repeat using different PT layers

---

### 10.2 Feature Subspace (Optional)
- Project hidden states onto PT_FT feature subspace
- Repeat experiment

---

### 10.3 Remove Shared Features (Optional)
- Ablate PT_FT features before decoding

---

## 11. Success Criteria

Evidence supporting the hypothesis requires:

1. **Linear map achieves low-distance matches**
   - Significantly better than all baselines

2. **Outputs resemble real time series**
   - Not noise-like

3. **Generalization**
   - Works on unseen text

4. **Structure preservation**
   - Captures periodicity / trends

---

## 12. Interpretation

### Positive Result

Implies:
- PT hidden states already lie in a space that overlaps with the **time series manifold**
- Language pretraining induces **domain-general sequential structure**

---

### Negative Result

Implies:
- Shared features are abstract but not sufficient for generation
- FT introduces genuinely new structure

---

## 13. Notes

- Keep decoder strictly linear to avoid overfitting
- Ensure evaluation uses held-out time series
- Carefully tune similarity weights \( \lambda_1, \lambda_2, \lambda_3 \)

---

## 14. Minimal Implementation Plan

1. Extract PT hidden states for WikiText
2. Load time series dataset
3. Implement linear layer (per timestep)
4. Implement similarity metrics (MSE + FFT + ACF)
5. Train with:
   - Variant 1 (soft)
   - Variant 2 (EM)
6. Evaluate retrieval + structure

---
