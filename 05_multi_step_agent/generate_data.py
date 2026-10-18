"""Generate two reproducible, entirely synthetic biomedical CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def create_dataset(n: int, seed: int, intercept: float, age_effect: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 10, n)
    sex = rng.binomial(1, 0.5, n)
    biomarker = intercept + age_effect * age + 12 * sex + rng.normal(0, 15, n)
    return pd.DataFrame({"age": age, "sex": sex, "biomarker": biomarker})


def main() -> None:
    specifications = {
        "biomarker": (200, 42, 100, 0.8),
        "biomarker_replication": (150, 2026, 90, 0.5),
    }
    for name, args in specifications.items():
        path = HERE / f"{name}.csv"
        data = create_dataset(*args)
        data.to_csv(path, index=False)
        print(f"Created {path.name} ({len(data)} rows)")
        print(data.head(3).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
