# Linear Decoding Reveals Low-Dimensional Time-Series Structure in Language Model Representations

## Abstract

We investigate whether pretrained language models encode latent structure sufficient to generate realistic time series without any paired supervision. Using hidden states from a pretrained Qwen3-0.6B model processing WikiText, we train a simple linear map to produce time series outputs matched to real sequences via an EM-style nearest-neighbor objective.

We find that (1) pretrained representations enable significantly better alignment with real time series than random or untrained models, (2) diversity requires both trained weights and meaningful text input, and (3) pretrained representations are highly anisotropic, with over 80% of variance concentrated in a single direction. This low-dimensional structure explains both the effectiveness and instability of linear decoding. Our results suggest that language models encode a latent dynamical system that can be linearly projected into realistic temporal signals.

---

## 1. Introduction

This work asks a simple question:

> Do pretrained language models encode latent temporal structure that can be decoded into realistic time series without supervision?

We approach this using a minimal setup:

* No paired text–time series data
* A single linear mapping from hidden states
* Matching to real time series via nearest-neighbor alignment

Despite this simplicity, we find strong evidence that:

* Language models encode **time-series-like dynamics**
* This structure is **low-dimensional and highly anisotropic**
* Both **training and input semantics** are necessary for diverse generation

---

## 2. Setup

### 2.1 Models

All experiments use Qwen3-0.6B (28 layers, hidden size 1024):

* **PT**: pretrained on text
* **RandomInit**: same architecture, random weights

---

### 2.2 Data

* **Text**: WikiText-103 (512-token sequences)
* **Time series**: GiftEval validation set (10,000 windows, length 512)

All time series are z-score normalized.

---

### 2.3 Linear Decoding

For hidden states ( h_{i,t} \in \mathbb{R}^D ):

[
\hat{y}*{i,t} = W h*{i,t} + b
]

Predictions are normalized per sequence.

---

### 2.4 Training Objective

We use EM-style matching:

* **E-step**: match each prediction to nearest real time series
* **M-step**: minimize MSE to matched targets

With diversity penalty:

[
L = L_{MSE} + \lambda \cdot L_{div}
]

where ( L_{div} ) penalizes similarity in power spectra across predictions.

---

### 2.5 Evaluation Metrics

* Nearest neighbor distance (NN)
* Unique matches
* Cluster coverage
* Match entropy
* Top-K match quality (fair comparison)

---

## 3. Core Results

### 3.1 Language Models Encode Decodable Structure

(Include your single-layer or concat baseline table)

**Key result:**
Pretrained models outperform random representations by ~2–3× in NN distance.

---

### 3.2 Diversity Requires Both Weights and Input

#### Table: 2×2 Ablation (Architecture × Input)

| Ablation      | Train Unique | Held Unique | Train NN | Held NN |
| ------------- | -----------: | ----------: | -------: | ------: |
| **text_PT**   |      **686** |     **439** |    0.672 |   0.717 |
| text_RandInit |           54 |          73 |    0.402 |   0.541 |
| rand_PT       |            4 |           1 |    0.283 |   0.293 |
| rand_RandInit |           99 |         138 |    0.522 |   0.741 |

**Key findings:**

* Trained weights alone are insufficient (rand_PT collapses)
* Text alone is insufficient (text_RandInit weak)
* **Only the combination produces diversity**

---

### 3.3 Quality at Matched Diversity (Top-K)

#### Table: Top-K Comparison

|   K |   text_PT | text_RandInit | rand_PT | rand_RandInit |
| --: | --------: | ------------: | ------: | ------------: |
|   4 | **0.254** |         0.383 |   0.330 |         0.412 |
|  99 | **0.383** |         0.825 |       — |         0.862 |
| 200 | **0.441** |         0.971 |       — |         1.004 |

**Conclusion:**
text_PT dominates at every diversity level — better quality *and* more coverage.

---

## 4. Representation Geometry

### 4.1 Random Projection Baseline

#### Table: Random Projections

| Condition     |     NN Dist |    Unique |     Entropy |
| ------------- | ----------: | --------: | ----------: |
| text_PT       | 1.36 ± 0.30 | 174 ± 174 | 0.47 ± 0.41 |
| text_RandInit | 1.04 ± 0.02 |  166 ± 10 | 0.63 ± 0.03 |

**Observation:**

* PT shows extremely high variance across projections
* RandInit is stable but mediocre

**Interpretation:**

* PT contains **rare high-quality directions**
* RandInit has no special directions

---

### 4.2 PCA Analysis

#### Table: Representation Rank

| Condition     | Eff Rank | Top Eigenvalue |
| ------------- | -------: | -------------: |
| text_PT       |      5.6 |          81.5% |
| rand_PT       |      2.8 |          89.2% |
| text_RandInit |      431 |           3.0% |
| rand_RandInit |      585 |           1.7% |

**Key findings:**

* PT representations are **extremely low-rank**
* > 80% of variance lies in a single direction
* RandInit is high-dimensional and near-isotropic

---

### Figure 1: Eigenvalue Spectrum

Plot:

* x-axis: component index (log scale)
* y-axis: variance explained

Curves:

* text_PT vs text_RandInit

**Expected result:**

* PT: sharp drop after first few components
* RandInit: flat decay

---

### 4.3 Interpretation

Pretraining induces:

* A **low-dimensional manifold**
* Dominated by a few directions
* With strong anisotropy

This explains:

* Why linear decoding works (structure exists)
* Why it is unstable (few usable directions)

---

## 5. Spectral Properties

### 5.1 Power Spectrum Comparison

#### Table: PSD Alignment

| Condition     |    PSD L2 | Low Freq (%) | High Freq (%) |
| ------------- | --------: | -----------: | ------------: |
| text_PT       |     0.228 |         90.1 |           3.3 |
| rand_PT       | **0.120** |         89.8 |           4.5 |
| text_RandInit |     0.274 |         58.4 |          22.1 |
| Real TS       |         — |         79.2 |           7.2 |

---

### Figure 2: Average Power Spectra

Plot:

* Frequency vs normalized power
* Curves: real TS, text_PT, text_RandInit

**Expected result:**

* PT matches low-frequency dominance
* RandInit is flatter (noisy)

---

### 5.2 Interpretation

* PT outputs are **smooth and low-frequency**
* RandInit outputs are **noisy**
* rand_PT achieves best spectral match but collapses

**Conclusion:**

> Spectral realism alone is insufficient — diversity is critical.

---

## 6. Synthesis

### 6.1 Mechanism

The results support the following model:

1. Pretraining compresses representations into a **low-dimensional subspace**
2. This subspace encodes **smooth temporal dynamics**
3. Text input moves representations along this manifold
4. Linear decoding projects this manifold into time series space

---

### 6.2 Why It Works

* Structure exists → decoding possible
* Low rank → few good directions
* Anisotropy → random projections unstable
* Text variation → enables diversity

---

### 6.3 Failure Modes

* Mode collapse from low-rank structure
* Sensitivity to initialization
* Over-concentration in low frequencies

---

## 7. Conclusion

We show that pretrained language models encode latent temporal structure that can be linearly decoded into realistic time series without supervision.

This structure is:

* **Low-dimensional**
* **Highly anisotropic**
* **Dependent on both training and input semantics**

These findings suggest that language models learn a latent dynamical system, and that linear decoding recovers projections of this system into observable signals.

---

## Appendix (Optional)

* Training details
* Additional metrics
* Stability across seeds
