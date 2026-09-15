"""Same-month cross-sectional median imputation and average-rank scaling.

Preprocess the FULL available monthly universe before selecting window samples.
Raw data and portfolio context are never overwritten.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.splits import validate_feature_columns


def load_factors(path):
    columns = pd.read_csv(Path(path))['variable'].tolist()
    validate_feature_columns(columns, columns)
    return columns


def monthly_rank_transform(panel, factors):
    validate_feature_columns(factors, factors)
    if not panel.index.is_unique:
        raise ValueError('panel index must be unique')
    if panel['eom'].isna().any() or panel.duplicated(['permno', 'eom']).any():
        raise ValueError('missing month or duplicate security-month')
    # Work one month at a time instead of materializing several 529k x147 arrays.
    result = np.empty((len(panel), len(factors)), dtype=np.float32)
    diagnostics = {'rows': len(panel), 'factors': len(factors),
                   'infinite_values_treated_as_missing': 0,
                   'missing_values_including_infinity': 0,
                   'all_missing_month_factor_pairs': 0, 'output_nonfinite': 0}
    for positions in panel.groupby('eom', sort=False).indices.values():
        values = panel.iloc[positions][factors].astype(float)
        diagnostics['infinite_values_treated_as_missing'] += int(np.isinf(values.to_numpy()).sum())
        values = values.replace([np.inf, -np.inf], np.nan)
        diagnostics['missing_values_including_infinity'] += int(values.isna().to_numpy().sum())
        diagnostics['all_missing_month_factor_pairs'] += int(values.isna().all().sum())
        filled = values.fillna(values.median()).fillna(0.0)
        if len(values) == 1:
            result[positions] = 0
        else:
            result[positions] = ((filled.rank(method='average') - 1) * (2 / (len(values) - 1)) - 1).to_numpy()
    if not np.isfinite(result).all():
        raise ValueError('non-finite preprocessed features')
    return pd.DataFrame(result, index=panel.index, columns=factors), diagnostics
