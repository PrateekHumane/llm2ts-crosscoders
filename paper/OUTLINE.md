# Paper Outline: The Geometric Prior
## Why Language Models Transfer to Time Series

---

### Title (working)
"The Geometric Prior: How Autoregressive Pretraining Creates Time-Series-Compatible Representations"

---

### Abstract
- Language models pretrained on text transfer surprisingly well to time series forecasting — but why?
- We argue this is explained by a **geometric prior**: autoregressive pretraining compresses hidden state trajectories onto a low-dimensional manifold whose dynamics naturally overlap with real-world temporal data.
- Three lines of evidence: (1) linear decoding from text hidden states produces realistic time series without paired supervision, (2) finetuning selects from pretrained representational directions rather than learning new ones, (3) individual neural features encoding temporal primitives (repetition, slopes, periodicity) transfer from language to time series.
- The connection is structural, not semantic: both domains share computational patterns that autoregressive models naturally learn to represent.

---

### 1. Introduction
- The puzzle: LLMs finetuned on TS match or beat domain-specific models. Why should text representations help with temporal data?
- Existing explanations are vague ("sequential structure" or "pattern recognition"). We provide a concrete mechanistic account.
- **Thesis**: Pretraining provides a geometric prior — a set of representational directions — that includes directions naturally suited for temporal data. Transfer = selecting those directions.
- Brief roadmap of the paper

### 2. Background
- LLM-to-TS transfer literature (brief)
- Sparse autoencoders / crosscoders for mechanistic interpretability
- Setup: PT, FT, RI models (same architecture, different training). GiftEval benchmark. WikiText-103.

### 3. The Geometric Prior Exists
*Claim: Pretrained hidden states already contain time-series-compatible structure, accessible via a single linear projection.*

- **Method**: Linear map from WikiText hidden states to TS, trained with EM-style nearest-neighbor matching (no paired supervision)
- **Result**: 686 unique matches out of 10,000 real TS, covering 14/20 temporal clusters
- **2×2 ablation** (architecture × input): disentangles what contributes
  - text+PT: 686 unique — needs both components
  - rand+PT: 4 unique — manifold exists but isn't traversed
  - text+RandInit: 54 unique — variation exists but no manifold
  - rand+RandInit: 99 unique — neither
- **Top-K fair comparison**: text+PT produces better quality at every diversity level
- **Interpretation**: Pretraining creates the manifold. Text traverses it. A linear map reads off the temporal signal.

### 4. The Prior is Geometric
*Claim: The manifold is low-dimensional and anisotropic. This is what makes linear decoding possible.*

- **PCA analysis**: text+PT has effective rank ~6 in 28,672 dimensions (81.5% of variance in one direction). RandInit has effective rank 431.
- **Spectral properties**: PT outputs are 90% low-frequency (matching real TS's 79%). RandInit is 58% (noisy).
- **Why this happens** (the math): Residual stream dynamics h_{t+1} = h_t + f(h_t) produce smooth trajectories. Pretraining compresses these to a low-rank subspace. Any linear projection of a smooth, low-rank trajectory produces a smooth 1D signal — i.e., a time series. The distribution of these projected signals overlaps with real TS distributions.
- **Connection to finetuning**: If the pretrained manifold already contains TS-like directions, then finetuning simply needs to identify which directions to use, not learn temporal dynamics from scratch.

### 5. Finetuning Selects, Doesn't Create
*Claim: FT reuses PT's pretrained directions rather than finding new ones. RI, lacking this prior, finds a different and less efficient solution.*

- **Subspace alignment**: ts@PT vs ts@FT = 60.6° (far below random baseline of 85.3°). ts@PT vs ts@RI = 73.7°.
- **Dimensional compression**: FT compresses from rank 44 → 12 (15 PCs for 90%). RI stays at rank 35 (46 PCs). FT found a more efficient encoding by leveraging the prior.
- **RI's clean geometry vs FT's complex reuse**: On periodic inputs (e.g. sine waves), RI discovers clean circular trajectories in hidden space — the minimal representation. FT produces topologically equivalent loops but geometrically more complex, using fewer PCA dimensions because it projects periodicity onto directions that were already high-variance from pretraining. FT's loops carry the "scars" of their linguistic origin. This is direct evidence of reuse rather than de novo learning.
- **Phase coherence**: FT represents periodicity most explicitly (lowest phase error: 0.08 vs PT 0.20, RI 0.13)
- **The cost**: Catastrophic forgetting as geometric compression — text→FT collapses to rank 2.5 (from 150). Trading ~140 text dimensions for ~12 TS dimensions.

### 6. What Transfers: Temporal Primitives
*Claim: Individual neural features encode specific temporal patterns that are shared between language and time series.*

- **Method**: Crosscoder (shared encoder, per-domain decoders) trained on 130k windows × 27 layers. WikiText probing with wiki-specific normalization.
- **Feature categorization** across layers: PT_FT features concentrated in early-to-mid layers
- **Discovered cross-domain features**:
  - Repetition detector (L8): "3·3·3·3" in math text ↔ square waves in TS
  - Slope detector (L26): "derivative", "tangent" ↔ rising/falling slopes
  - Metronome/beat detector (L4): "tempo", "beats" ↔ periodic spikes
  - Chord progression detector (L5): harmonic sequences ↔ periodic peaks
- **Pure TS features** (FT_RI): spike detectors, trough detectors, oscillation detectors — what both models converge on independently
- **Interpretation**: The transferred features are **structural primitives** — repetition, gradients, periodicity — not semantic concepts. These are computational building blocks useful for any sequential prediction task.

### 7. Discussion
- **The full picture**: The geometric prior is created by autoregressive dynamics (smooth trajectories) + pretraining compression (low rank) + implicit temporal feature learning (primitives). Finetuning exploits this by selecting useful directions and amplifying temporal features.
- **Why structural, not semantic**: Features detect patterns (repetition, slopes) not domain-specific knowledge. The shared manifold arises from computational overlap, not world knowledge.
- **Implications**: This suggests transfer should work for any pretrained autoregressive model, not just language models. The key property is smooth, low-dimensional sequential representations.
- **Limitations**: Single architecture, single benchmark, high dead feature rate in crosscoders, crosscoder features may miss subtle connections.

### 8. Conclusion
- Language-to-TS transfer is explained by a geometric prior: low-dimensional, smooth, anisotropic representations that happen to share statistical structure with temporal data.
- Finetuning selects from this prior; it doesn't build temporal representations from scratch.
- The transfer is structural: both domains rely on the same computational primitives (repetition, slopes, periodicity).
