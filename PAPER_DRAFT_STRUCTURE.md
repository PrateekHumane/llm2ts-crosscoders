# The Shared Manifold: Why Language Models Transfer to Time Series
## NeurIPS Paper Structure — Draft v1

---

## Working Title Options
1. "The Shared Manifold: How Autoregressive Pretraining Creates Time-Series-Compatible Representations"
2. "From Words to Waves: A Mechanistic Account of Language-to-Time-Series Transfer"
3. "Geometry, Features, and Directions: Why Language Models Encode Temporal Structure"

---

## Abstract (Draft)

Language models pretrained on text can be finetuned to forecast time series, often matching domain-specific architectures—but why? We present a mechanistic account grounded in three complementary lines of evidence. First, we show that a single linear projection of a pretrained Qwen3-0.6B model's hidden states—processing ordinary Wikipedia text with no paired supervision—produces realistic time series matching 686 of 10,000 real signals across 14 temporal clusters. A 2×2 ablation (architecture × input) confirms this requires both pretrained weights and meaningful text. Second, geometric analysis of hidden state trajectories reveals the mechanism: pretraining compresses representations onto a ~6-dimensional manifold; finetuning on time series further collapses to ~12 dimensions by selecting from the pretrained directions (60.6° subspace alignment vs. 85.3° random baseline). Third, crosscoder feature analysis across 27 layers identifies specific shared neural features—a repetition detector that fires on "3·3·3·3" in text and square waves in time series, a slope detector that fires on the word "derivative" and on rising/falling temporal signals. Together, these results show that language-to-time-series transfer is not a mysterious emergent capability but a consequence of geometry: autoregressive pretraining creates smooth, low-dimensional trajectories that share statistical structure with temporal data, and finetuning selectively amplifies the pretrained directions that encode this structure.

---

## Paper Structure

### 1. Introduction
- The puzzle: LLMs transfer to TS despite no exposure to temporal data
- Recent work on LLM-to-TS transfer (brief lit review)
- Our contribution: THREE complementary experiments that together explain WHY
  - Linear decoding (shows structure exists)
  - Representational geometry (shows the mechanism)  
  - Crosscoder features (shows what transfers at the neuron level)
- Key claim: **Shared Manifold Hypothesis** — autoregressive pretraining creates a low-dimensional manifold whose temporal dynamics align with real-world time series

### 2. Background and Setup
- Models: PT, FT, RI (same architecture, different training)
- Data: GiftEval time series, WikiText-103 text
- Tokenization: uniform binning for FT/RI, text serialization with mean-pooling for PT

### 3. Experiment 1: Linear Decoding Reveals Latent Temporal Structure
**Question**: Do pretrained hidden states contain time-series-decodable structure?

- Method: Linear projection from hidden states to TS, EM-style matching
- **Result**: text+PT produces 686 unique matches (14/20 clusters)
- **2×2 Ablation**: 
  - text+PT: 686 unique (needs BOTH weights and text)
  - rand+PT: 4 unique (manifold exists but not traversed)
  - text+RandInit: 54 unique (text varies but no manifold)
  - rand+RandInit: 99 unique (neither)
- **Fair Top-K comparison**: text+PT dominates at every diversity level
- **Layer sweep**: Peak at L2-L12 (3×), transition at L16, recovery at L20+
- Key insight: **Structure requires pretraining; diversity requires meaningful input**

### 4. Experiment 2: Representational Geometry Explains the Mechanism
**Question**: What geometric properties enable transfer?

#### 4.1 The Pretrained Manifold is Low-Dimensional
- PCA: text+PT has effective rank 5.6 (81.5% in first eigenvalue)
- RandInit: effective rank 431 (isotropic)
- This explains why linear decoding works: few directions, all structured

#### 4.2 Finetuning Selects from Pretrained Directions
- TS→FT: eff rank 12, 15 PCs for 90% (compressed from PT's 44)
- TS→RI: eff rank 35, 46 PCs (higher dimensional, different subspace)
- **Subspace alignment**: ts@PT vs ts@FT = 60.6° (random baseline 85.3°)
- ts@PT vs ts@RI = 73.7° (much less aligned)
- **Interpretation**: FT reuses PT's directions, not new ones

#### 4.3 Spectral Properties
- PT outputs: 90.1% low-frequency energy (matches real TS's 79.2%)
- RandInit: 58.4% (noisy)
- FT: 52.3% (smoothest)

#### 4.4 Catastrophic Forgetting as Geometric Compression
- Text→FT collapses to rank 2.5 (from 150): traded ~140 text dimensions for ~12 TS dimensions
- This is the geometric cost of transfer

#### 4.5 Manifold Structure
- Periodic inputs → looping trajectories in all models
- FT: tightest phase-coherent loops (phase error 0.08 vs PT's 0.20)
- RI: simpler, more symmetric loops (learned from scratch)

### 5. Experiment 3: Crosscoder Feature Analysis Reveals What Transfers
**Question**: What specific features are shared between text and time series?

#### 5.1 Crosscoder Architecture
- Shared encoder (1024→4096, TopK=64), per-domain decoders
- Each domain encoded independently
- Trained on 130k windows per layer, 27 layers

#### 5.2 Feature Categorization
- Category breakdown across layers (stacked bar chart)
- PT_FT features (language transfer) concentrated in early-to-mid layers
- FT_RI features (convergent learning) dominate

#### 5.3 WikiText Probing
- Key methodological point: wiki-specific normalization required
- Features fire on specific semantic tokens in text

#### 5.4 Discovered Cross-Domain Features
The headline findings:
1. **Repetition detector** (L8, score 8/10): "3·3·3·3" in math text ↔ square waves in TS
2. **Slope detector** (L26, score 7/10): "derivative", "tangent line" ↔ rising/falling slopes
3. **Metronome detector** (L4, score 6/10): "beats", "tempo" ↔ periodic spikes
4. **Chord progression detector** (L5, score 6/10): "F-E♭-G" ↔ periodic peaks
5. **Sonata detector** (L6, score 6/10): musical structure ↔ sinusoidal oscillations

#### 5.5 Pure Time Series Features (FT_RI)
- Periodic trough detector (L6, score 8)
- High-frequency oscillation detector (L7, score 8)
- Spike detector (L26, score 8)
- These show what FT and RI learn convergently

### 6. Synthesis: The Three Lines of Evidence
- **Linear decoding** shows the structure exists and can be accessed with a single direction
- **Geometry** shows the structure is a low-dimensional manifold created by pretraining, from which FT selects useful directions
- **Features** show that the transfer operates at the level of individual neurons detecting specific temporal patterns (repetition, slopes, periodicity) that are useful in both language and time series
- Together: transfer is not mysterious — it's geometry + features

### 7. Discussion
- The connection is primarily **structural**, not semantic
  - Features detect computational patterns (repetition, slopes) not domain-specific semantics
  - The shared manifold arises from autoregressive dynamics, not world knowledge
- Implications for TS foundation models
- Limitations:
  - Single architecture (Qwen3-0.6B)
  - Single TS benchmark (GiftEval)  
  - Crosscoder features are noisy (many dead features, 40-60% per layer)
- The cost of transfer: catastrophic forgetting (Text→FT rank 2.5)

### 8. Conclusion

---

## Key Figures (Proposed)

1. **Fig 1**: Hero figure — three-panel showing (a) linear decoded TS from WikiText, (b) 2×2 ablation table, (c) crosscoder feature example (repetition detector with text + TS)
2. **Fig 2**: Representation geometry — eigenvalue spectra (PT sharp drop vs RandInit flat)
3. **Fig 3**: Subspace alignment across layers — ts@PT↔ts@FT stays far below random baseline
4. **Fig 4**: Manifold structure — 2D PCA of trajectories through PT/FT/RI with explained variance
5. **Fig 5**: Category breakdown stacked bar chart across 27 layers
6. **Fig 6**: Top crosscoder features — 3-4 panels showing TS windows + WikiText spans side by side
7. **Fig 7**: Training loss per layer + convergence detection

---

## Key Tables

1. **Table 1**: 2×2 ablation (unique matches, NN distance, clusters)
2. **Table 2**: Top-K fair comparison
3. **Table 3**: Representation geometry (eff rank, top eigenvalue, PCs for 90%)
4. **Table 4**: Subspace alignment (principal angles between conditions)
5. **Table 5**: Top crosscoder features with scores and interpretations

---

## Potential Concerns / Reviewer Questions

1. **"The linear decoding is just finding smooth signals, not real TS structure"**
   → Top-K comparison shows text+PT has better quality at matched diversity, not just smooth noise
   → Spectral analysis confirms match to real TS frequency profile
   
2. **"The crosscoder features could be coincidental"**
   → We scored 2,430 features; only 16 scored 5+. We're honest about the hit rate.
   → The strongest features (repetition, slope) are genuinely compelling
   
3. **"Single model, single dataset"**
   → Fair limitation. But the geometric analysis is architecture-general (smooth residual streams)
   
4. **"Dead features (40-60%)"**
   → Standard for TopK SAEs. The alive features are informative.

5. **"The wiki normalization fix — does it invalidate the categorization?"**
   → No, categorization used TS activations (domain-specific norm). Wiki probing is post-hoc.
