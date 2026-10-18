"""Statistical diagnostics kept independent of the LLM."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan


def check_ols_diagnostics(df: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    """Compute basic diagnostics for a restricted additive OLS model.

    These are screening tools, NOT a guarantee that OLS assumptions hold.
    """
    if outcome != "biomarker" or not predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Outcome must be 'biomarker'; predictors must be distinct and nonempty")
    if any(p not in {"age", "sex"} for p in predictors):
        raise ValueError("Only 'age' and 'sex' are supported predictors")
    cols = [outcome] + predictors
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing required columns: {set(cols)-set(df.columns)}")
    data = df[cols].dropna()
    if len(data) <= len(predictors) + 2:
        raise ValueError("Not enough complete observations for diagnostics")
    formula = outcome + " ~ " + " + ".join(predictors)
    model = smf.ols(formula, data=data).fit()
    resid = np.asarray(model.resid)
    shapiro = stats.shapiro(resid) if len(resid) <= 5000 else None
    bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(resid, model.model.exog)
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    return {
        "formula": formula,
        "n_observations": int(model.nobs),
        "shapiro_wilk_p": float(shapiro.pvalue) if shapiro is not None else None,
        "breusch_pagan_lm_p": float(bp_p),
        "breusch_pagan_f_p": float(bp_f_p),
        "max_cooks_distance": float(np.max(cooks)),
        "n_cooks_distance_above_4_over_n": int(np.sum(cooks > 4 / len(data))),
        "cooks_reference_threshold_4_over_n": float(4 / len(data)),
        "notes": [
            "Shapiro-Wilk assesses departures from normality, not model linearity.",
            "Breusch-Pagan screens for particular heteroscedasticity patterns; p-values do not prove constant variance.",
            "Cook's distance identifies potentially influential cases; 4/n is a heuristic, not a definitive cutoff.",
            "Independence, causal identification, and functional form are not established by these statistics. Inspect plots and research design."
        ],
    }
