# Mapping Experiment Plan

## Completed Experiments

### Experiment 1: Basic Linear Mapping (Layer 8)
- **Status**: DONE
- **Result**: PT=0.72, RandomInit=0.79, Random=1.69, Shuffled=1.77
- **Finding**: PT 2.4x better than Random. Shuffled ≈ Random (temporal order matters).

### Experiment 2: Layer Sweep (PT vs Random)
- **Status**: DONE
- **Result**: PT beats Random 2.2-3.1x at all layers. Peak at L2-L12.

### Experiment 3: Random-Init Control
- **Status**: DONE
- **Result**: Architecture alone gives 2.5x over Random. Language training adds 15-20% at early layers, hurts at L16+.

### Experiment 4: Diversity Analysis
- **Status**: DONE
- **Finding**: Mode collapse — PT matches 71% of predictions to top 5 TS. RandomInit matches 98% to top 5. NN distance metric is misleading.

## Current Experiments (To Run)

### Experiment B: Diversity-Penalized Training
- **Goal**: Force diverse predictions by penalizing similar outputs within a batch
- **Penalties**: 
  - PSD diversity: penalize similar frequency spectra (shift-invariant)
  - ACF diversity: penalize similar autocorrelation profiles (shift-invariant)
- **Lambda sweep**: 0.1, 0.5, 1.0
- **Models**: PT, RandomInit, Random at layer 8
- **Script**: `scripts/mapping_diversity.py`

### Experiment A: Cluster Evaluation
- **Goal**: Evaluate how many different types of TS each model can approximate
- **Method**: Cluster real TS into 20 groups by ACF profile, measure per-cluster distance and coverage
- **Metrics**: cluster coverage, per-cluster distance, reverse coverage, match entropy
- **Run alongside Experiment B** (same script)

## Future Experiments (If Needed)

### Experiment C: Small Bottleneck
- **Goal**: Give model capacity for diversity without overfitting
- **Method**: Replace 1024→1 linear with 1024→8→1 bottleneck (8 directions)
- **Rationale**: Single linear direction forces mode collapse; 8 directions allow diverse outputs

### Experiment D: Input Variance Analysis  
- **Goal**: Check if the model is actually using the text input
- **Method**: Measure variance of predictions across different WikiText inputs
- **If variance ≈ 0**: model ignores input, just learned a fixed output

### Experiment E: Multi-Metric Matching
- **Goal**: Use MSE + PSD + ACF for finding nearest TS (not just MSE)
- **Rationale**: May find better structural matches

### Experiment F: Layer Sweep with Diversity
- **Goal**: Repeat layer sweep with diversity penalty, see if the layer profile changes
- **Depends on**: Experiment B results

### Experiment G: Reverse Mapping — What Text Looks Like a Given Time Series?

Two sub-experiments exploring the text↔TS correspondence from the TS side.

#### G1: Text retrieval via trained mapper (PT hidden states)
- **Goal**: For representative TS from distinct domains, find WikiText passages whose PT hidden states decode (via trained W) into the closest match.
- **Method**:
  1. Select ~10 target TS from distinct GiftEval domains (electricity, solar, traffic, weather, covid, births, hydrology, restaurant, hospital, cloud). Pick the most "typical" window per domain (closest to cluster centroid).
  2. Use saved predictions from 1920 WikiText training sequences (already projected through W).
  3. For each target TS, find top-5 WikiText sequences by MSE between prediction and target.
  4. Retrieve and display the actual text of those passages alongside the TS overlay.
- **Key question**: Is there any semantic connection (temporal language?), or is it purely geometric?
- **Status**: IN PROGRESS

#### G2: TS through FT/RI — hidden state matching to text
- **Goal**: Pass TS through FT and RI models (which understand TS tokens), get hidden states, and find which text token sequences produce the most similar hidden states in the same model.
- **Method**: TBD after G1 results.
- **Status**: PLANNED
