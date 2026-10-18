"""Pure statistical computations, independent of any LLM framework."""

import numpy as np


def calculate_statistics(values: list[float]) -> dict:
    """Return basic summary statistics for finite numerical observations."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Provide at least two numerical observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("All observations must be finite numbers.")
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sample_sd": float(np.std(x, ddof=1)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


if __name__ == "__main__":
    print(calculate_statistics([12, 15, 18, 20, 25]))
