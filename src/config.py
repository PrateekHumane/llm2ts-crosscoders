from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Models ---
    model_ft: str = "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-8192"
    model_ri: str = "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-8192"
    model_pt: str = "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # --- Tokenizer (FT/RI uniform binning) ---
    n_bins: int = 1024
    bin_low: float = -5.0
    bin_high: float = 5.0
    n_special_tokens: int = 2

    # --- Data ---
    dataset_name: str = "Salesforce/GiftEval"
    context_length: int = 512   # from training_config.json
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test = remaining 0.15

    # --- Model architecture ---
    hidden_size: int = 1024
    num_layers: int = 28

    # --- Crosscoder ---
    latent_dim: int = 4096
    top_k: int = 64
    mlp_hidden: int = 2048
    linear_crosscoder: bool = True

    # --- Training ---
    batch_size: int = 64       # windows per batch
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 100_000
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    weight_decay: float = 0.0
    dead_neuron_window: int = 1000
    dead_neuron_threshold: float = 0.01
    auxk_coeff: float = 1 / 32      # weight for auxiliary (dead-feature) loss
    auxk_k: int = 64                 # top-k among dead features for aux loss
    auxk_start_step: int = 1000      # begin aux loss after dead counters stabilise

    # --- Multi-GPU ---
    # 28 layers split across 4 GPUs → 7 per GPU
    num_gpus: int = 4
    layers_per_gpu: int = 7

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    analysis_dir: str = "analysis"

    # --- Dtype ---
    model_dtype: str = "bfloat16"   # for the three transformer models
    crosscoder_dtype: str = "float32"

    # --- Precomputation ---
    # Number of training windows to precompute activations for.
    # One layer at a time: 130k × 512 × 1024 × 3 × 2 bytes ≈ 409 GB (fits in 478 GB disk).
    n_precompute_windows: int = 130_000
    precompute_dir: str = "precomputed_acts"   # root dir for saved activations

    # --- Feature extraction ---
    top_n_windows: int = 100        # top activating windows per feature
