"""2026 adaptation of the supplied penalized_linear_hackathon.py.

Preserves dense ranks, train-only standardization, demeaned train targets,
no-intercept estimators, and the original four models and alpha grids. Fixes
label-dependent prediction membership and delegates dates to member C.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.exceptions import ConvergenceWarning
from src.data.build_samples import KEYS, annual_indices
from src.data.splits import assert_fit_scope, validate_feature_columns
from src.training.metrics import prediction_metrics


def dense_monthly_transform(panel, factors):
    validate_feature_columns(factors, factors)
    if not panel.index.is_unique or panel.duplicated(['permno', 'eom']).any() or panel.eom.isna().any():
        raise ValueError('invalid monthly panel keys')
    values = panel[factors].astype(float).replace([np.inf, -np.inf], np.nan)
    group = values.groupby(panel.eom, sort=False)
    filled = values.fillna(group.transform('median'))
    ranks = filled.groupby(panel.eom, sort=False).rank(method='dense') - 1
    maximum = ranks.groupby(panel.eom, sort=False).transform('max')
    result = (ranks / maximum.replace(0, np.nan) * 2 - 1).fillna(0.0)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('non-finite transformed values')
    return result, {'rows': len(panel), 'factors': len(factors),
                    'missing_including_infinity': int(values.isna().to_numpy().sum()),
                    'all_missing_month_factor_pairs': int(group.count().eq(0).to_numpy().sum()),
                    'output_nonfinite': 0}


def official_grids():
    return {'ols': [None], 'lasso': list(10.0 ** np.arange(-4, 4.1, .1)),
            'ridge': list(.5 * 10.0 ** np.arange(-1, 8.1, .1)),
            'en': list(10.0 ** np.arange(-4, 4.1, .1))}


def estimator(name, alpha):
    if name == 'ols': return LinearRegression(fit_intercept=False)
    if name == 'ridge': return Ridge(alpha=alpha, fit_intercept=False)
    if name == 'lasso': return Lasso(alpha=alpha, max_iter=1000000, fit_intercept=False)
    if name == 'en': return ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=1000000, fit_intercept=False)
    raise ValueError('unknown estimator')


def fit_official_year(panel, features, factors, year, grids=None):
    validate_feature_columns(list(features.columns), factors)
    if list(features.columns) != list(factors) or not panel.index.equals(features.index):
        raise ValueError('feature order or index mismatch')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('non-finite features')
    grids = official_grids() if grids is None else grids
    if set(grids) != {'ols', 'lasso', 'ridge', 'en'}:
        raise ValueError('expected all four models')
    for name, grid in grids.items():
        if not len(grid) or (name == 'ols' and list(grid) != [None]):
            raise ValueError('invalid grid')
        if name != 'ols' and any(not np.isfinite(a) or a <= 0 for a in grid):
            raise ValueError('invalid alpha')
    partitions = annual_indices(panel, year)
    labels = panel.ret_exc_lead1m.to_numpy(dtype=float)
    train = partitions['train'][np.isfinite(labels[partitions['train']])]
    val = partitions['validation'][np.isfinite(labels[partitions['validation']])]
    test = partitions['test']
    if not len(train) or not len(val) or not len(test):
        raise ValueError('empty partition')
    assert_fit_scope(panel.loc[train, KEYS].to_dict('records'), year)
    scaler = StandardScaler().fit(features.loc[train])
    xt, xv, xs = (scaler.transform(features.loc[ids]) for ids in [train, val, test])
    ymean = float(labels[train].mean())
    centered = labels[train] - ymean
    output = panel.loc[test, KEYS + ['ret_exc_lead1m']].copy()
    output['label_available'] = np.isfinite(labels[test])
    output['model_year'] = year
    report = {'year': year, 'train_rows': len(train), 'validation_rows': len(val),
              'test_rows': len(test), 'models': {}}
    parameters = {'feature_order': factors, 'scaler_mean': scaler.mean_.tolist(),
                  'scaler_scale': scaler.scale_.tolist(), 'target_train_mean': ymean, 'models': {}}
    for name, grid in grids.items():
        trials, best_mse, best_model, selected_alpha = [], float('inf'), None, None
        for alpha in grid:
            model = estimator(name, alpha)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(xt, centered)
            validation = prediction_metrics(labels[val], model.predict(xv) + ymean)
            messages = [str(w.message) for w in caught]
            trials.append({'alpha': alpha, 'validation_mse': validation['mse'], 'warnings': messages})
            if validation['mse'] < best_mse:
                best_model, best_mse, selected_alpha = model, validation['mse'], alpha
        # Same train-only estimator as refitting selected alpha, without duplicate work.
        output[name] = best_model.predict(xs) + ymean
        report['models'][name] = {'selected_alpha': selected_alpha, 'validation_trials': trials,
                                 'test': prediction_metrics(labels[test], output[name])}
        parameters['models'][name] = {'alpha': selected_alpha, 'coef': best_model.coef_.tolist(),
                                     'fit_intercept': False, 'l1_ratio': .5 if name == 'en' else None}
    return output, report, parameters
