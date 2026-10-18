"""Deterministic, LLM-independent statistics. Only named bundled datasets are accessible."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

DATA = Path(__file__).resolve().parents[1] / 'data'
ALLOWED = {'biomarker', 'biomarker_replication'}

def load_dataset(dataset: str) -> pd.DataFrame:
    if dataset not in ALLOWED:
        raise ValueError(f'Unknown dataset {dataset!r}; choose from {sorted(ALLOWED)}')
    return pd.read_csv(DATA / f'{dataset}.csv')

def inspect_dataset(dataset: str) -> dict:
    df = load_dataset(dataset)
    return {'dataset': dataset, 'rows': len(df), 'columns': df.columns.tolist(),
            'missing': {k:int(v) for k,v in df.isna().sum().items()}}

def describe_numeric(dataset: str, column: str) -> dict:
    df = load_dataset(dataset)
    if column not in df: raise ValueError(f'Unknown column: {column}')
    if not pd.api.types.is_numeric_dtype(df[column]): raise ValueError('Numeric column required')
    x = df[column].dropna()
    if x.empty: raise ValueError('No nonmissing data')
    return {'dataset':dataset,'column':column,'n':len(x),'mean':float(x.mean()),
            'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,
            'min':float(x.min()),'max':float(x.max())}

def regression(dataset: str, outcome: str, predictors: list[str]) -> dict:
    df = load_dataset(dataset)
    if not predictors or len(predictors) != len(set(predictors)) or outcome in predictors:
        raise ValueError('Supply distinct predictors, excluding outcome')
    columns=[outcome]+predictors
    if any(c not in df for c in columns): raise ValueError(f'Unknown columns; available {df.columns.tolist()}')
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in columns): raise ValueError('Numeric variables required')
    sample=df[columns].dropna()
    if len(sample) <= len(predictors)+1: raise ValueError('Too few complete observations')
    X=sm.add_constant(sample[predictors],has_constant='add')
    fit=sm.OLS(sample[outcome],X).fit()
    return {'dataset':dataset,'outcome':outcome,'predictors':predictors,'n':len(sample),
            'coefficients':{k:float(v) for k,v in fit.params.items()},
            'p_values':{k:float(v) for k,v in fit.pvalues.items()}, 'r_squared':float(fit.rsquared)}

def regression_diagnostics(dataset: str, outcome: str, predictors: list[str]) -> dict:
    df=load_dataset(dataset)
    regression(dataset,outcome,predictors)  # validates specification
    sample=df[[outcome]+predictors].dropna()
    fit=sm.OLS(sample[outcome],sm.add_constant(sample[predictors],has_constant='add')).fit()
    p_shapiro=float(shapiro(fit.resid).pvalue) if len(sample) <= 5000 else None
    p_bp=float(het_breuschpagan(fit.resid,fit.model.exog)[1])
    cooks=fit.get_influence().cooks_distance[0]
    return {'n':len(sample),'shapiro_p':p_shapiro,'breusch_pagan_p':p_bp,
            'max_cooks_distance':float(np.max(cooks)),
            'note':'Diagnostic tests are not proofs that assumptions hold.'}
