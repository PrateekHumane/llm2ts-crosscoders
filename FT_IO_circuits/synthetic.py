"""Synthetic time series generators for circuit analysis."""
import numpy as np


def generate_sine(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.arange(T, dtype=np.float64)
    for i in range(n):
        freq = rng.uniform(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        out[i] = np.sin(2 * np.pi * freq * t / T + phase)
    return out


def generate_linear_trend(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.linspace(0, 1, T)
    for i in range(n):
        slope = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 3.0)
        out[i] = slope * t + rng.normal(0, 0.05, T)
    return out


def generate_step(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    for i in range(n):
        n_steps = rng.integers(1, 4)
        positions = sorted(rng.integers(T // 6, 5 * T // 6, size=n_steps))
        level = 0.0
        prev = 0
        for p in positions:
            out[i, prev:p] = level
            level += rng.uniform(-2, 2)
            prev = p
        out[i, prev:] = level
    return out


def generate_exponential(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.linspace(0, 1, T)
    for i in range(n):
        rate = rng.uniform(1.0, 4.0)
        if rng.random() < 0.5:
            out[i] = np.exp(rate * t) - 1
        else:
            out[i] = np.exp(-rate * t)
    return out


def generate_sawtooth(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.arange(T, dtype=np.float64)
    for i in range(n):
        period = rng.integers(30, 200)
        out[i] = (t % period) / period
    return out


def generate_random_walk(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    for i in range(n):
        step_std = rng.uniform(0.05, 0.2)
        out[i] = np.cumsum(rng.normal(0, step_std, T))
    return out


def generate_ar1(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    for i in range(n):
        phi = rng.uniform(0.85, 0.99)
        noise_std = rng.uniform(0.1, 0.5)
        x = 0.0
        for j in range(T):
            x = phi * x + rng.normal(0, noise_std)
            out[i, j] = x
    return out


def generate_seasonal(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.arange(T, dtype=np.float64)
    for i in range(n):
        for _ in range(rng.integers(2, 4)):
            freq = rng.uniform(1, 15)
            amp = rng.uniform(0.3, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            out[i] += amp * np.sin(2 * np.pi * freq * t / T + phase)
    return out


def generate_constant(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    for i in range(n):
        out[i] = rng.uniform(-2, 2) + rng.normal(0, 0.01, T)
    return out


def generate_white_noise(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    for i in range(n):
        out[i] = rng.normal(0, rng.uniform(0.5, 2.0), T)
    return out


def generate_square_wave(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.arange(T, dtype=np.float64)
    for i in range(n):
        period = rng.integers(20, 150)
        duty = rng.uniform(0.3, 0.7)
        out[i] = np.where((t % period) / period < duty, 1.0, -1.0)
    return out


def generate_damped_sine(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, T))
    t = np.arange(T, dtype=np.float64)
    for i in range(n):
        freq = rng.uniform(3, 10)
        decay = rng.uniform(2, 8)
        phase = rng.uniform(0, 2 * np.pi)
        out[i] = np.exp(-decay * t / T) * np.sin(2 * np.pi * freq * t / T + phase)
    return out


GENERATORS = {
    "sine": generate_sine,
    "linear_trend": generate_linear_trend,
    "step": generate_step,
    "exponential": generate_exponential,
    "sawtooth": generate_sawtooth,
    "random_walk": generate_random_walk,
    "ar1": generate_ar1,
    "seasonal": generate_seasonal,
    "constant": generate_constant,
    "white_noise": generate_white_noise,
    "square_wave": generate_square_wave,
    "damped_sine": generate_damped_sine,
}


def generate_all(n_per_type: int = 50, context_length: int = 512,
                 seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: gen(n_per_type, context_length, rng)
            for name, gen in GENERATORS.items()}


def tokenize_windows(windows: np.ndarray, n_bins: int = 1024,
                     low: float = -5.0, high: float = 5.0) -> np.ndarray:
    """Normalize per-window and bin-tokenize. Returns (N, T) int64 in [0, n_bins-1]."""
    N, T = windows.shape
    tokens = np.zeros((N, T), dtype=np.int64)
    for i in range(N):
        mu, sigma = windows[i].mean(), windows[i].std()
        if sigma < 1e-8:
            sigma = 1.0
        normed = (windows[i] - mu) / sigma
        clipped = np.clip(normed, low, high)
        bins = ((clipped - low) / (high - low) * n_bins).astype(np.int64)
        tokens[i] = np.clip(bins, 0, n_bins - 1)
    return tokens
