"""Train-only Ridge coefficients, validation-only alpha selection."""
import numpy as np
from sklearn.linear_model import Ridge
from src.data.build_samples import KEYS, annual_indices
from src.data.splits import assert_fit_scope, validate_feature_columns
from src.training.metrics import prediction_metrics


def fit_year(panel, features, factors, year, alphas):
    validate_feature_columns(list(features.columns), factors)
    if list(features.columns) != list(factors) or not features.index.equals(panel.index):
        raise ValueError('feature order/index mismatch')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('non-finite features')
    alphas = sorted(set(float(a) for a in alphas))
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('alphas must be positive finite values')
    partitions = annual_indices(panel, year)
    y = panel['ret_exc_lead1m'].to_numpy(dtype=float)
    supervised = {name: ids[np.isfinite(y[ids])] for name, ids in partitions.items()}
    train, validation = supervised['train'], supervised['validation']
    if not len(train) or not len(validation) or not len(partitions['test']):
        raise ValueError('empty train, validation or prediction partition')
    assert_fit_scope(panel.loc[train, KEYS].to_dict('records'), year)
    trials, best, best_loss = [], None, float('inf')
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver='svd')
        model.fit(features.loc[train], y[train])
        metrics = prediction_metrics(y[validation], model.predict(features.loc[validation]))
        trials.append({'alpha': alpha, **metrics})
        if metrics['mse'] < best_loss:
            best, best_loss = model, metrics['mse']
    test = partitions['test']
    prediction = best.predict(features.loc[test])
    output = panel.loc[test, KEYS + ['ret_exc_lead1m']].copy()
    output['prediction'] = prediction
    output['label_available'] = np.isfinite(y[test])
    output['model_year'] = year
    output['alpha'] = best.alpha
    result = {'year': year, 'selected_alpha': best.alpha, 'validation_trials': trials,
              'test_metrics': prediction_metrics(y[test], prediction),
              'partitions': {name: {'rows': len(ids), 'supervised_rows': len(supervised[name]),
                   'first_target': str(panel.loc[ids, 'target_month'].min()) if len(ids) else None,
                   'last_target': str(panel.loc[ids, 'target_month'].max()) if len(ids) else None}
                   for name, ids in partitions.items()}}
    return output, result, best
