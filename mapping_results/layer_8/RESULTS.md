# Mapping Experiment Results — Layer 8

## NN Distance (lower = predicted TS is closer to real TS)

| Method | NN Mean | NN Median |
|--------|---------|-----------|
| RI     | 0.4629  | 0.4655    |
| **PT** | **0.7161** | **0.7132** |
| Random | 1.6873  | 1.6928    |
| Shuffled | 1.7729 | 1.7812   |

## Key Findings

1. PT >> Random (2.4x better): Language model hidden states from WikiText contain time-series-relevant structure
2. Shuffled ≈ Random: Temporal order in PT matters — it's the sequential processing, not just weights
3. RI > PT: RI's time-series-specialized weights produce better structure even from text tokens
4. All trained models >> Random: Both PT and RI produce structured outputs from text

## Interpretation

PT's hidden states when processing natural language text already contain enough sequential structure 
that a single linear projection per timestep can produce outputs resembling real time series.
This structure is destroyed by shuffling (proving it's temporal, not just statistical).
