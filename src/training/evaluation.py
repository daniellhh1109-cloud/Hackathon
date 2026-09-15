"""Prediction-only evaluation: zero forecast R², Huber and monthly rank IC.

No portfolio/financial-return performance is implied by these statistics.
"""
import numpy as np
import pandas as pd
from src.training.metrics import prediction_metrics


def regression_metrics(y, prediction, huber_delta=1.0):
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError('positive finite Huber delta required')
    result = prediction_metrics(y, prediction)
    y, prediction = np.asarray(y,dtype=float), np.asarray(prediction,dtype=float)
    usable = np.isfinite(y)
    error = np.abs(y[usable]-prediction[usable])
    quadratic = np.minimum(error,huber_delta)
    huber = .5*quadratic**2 + huber_delta*(error-quadratic)
    result.update({'mae':float(error.mean()) if len(error) else None,
                   'rmse':float(np.sqrt(result['mse'])) if len(error) else None,
                   'huber_loss':float(huber.mean()) if len(error) else None,
                   'huber_delta':huber_delta,
                   'huber_linear_fraction':float((error>huber_delta).mean()) if len(error) else None})
    return result


def evaluate_table(table, model_columns, huber_delta=1.0):
    """All models must predict every row; score all against the same finite labels."""
    required={'permno','target_month','realized_target',*model_columns}
    if not required <= set(table):
        raise ValueError('missing evaluation columns')
    if table.empty or table[['permno','target_month']].isna().any().any() or table.duplicated(['permno','target_month']).any():
        raise ValueError('empty/invalid/duplicate evaluation keys')
    summaries,monthly=[],[]
    for name in model_columns:
        result=regression_metrics(table.realized_target,table[name],huber_delta)
        ics=[]
        for month,group in table.groupby('target_month',sort=True):
            metrics=regression_metrics(group.realized_target,group[name],huber_delta)
            finite=group[np.isfinite(group.realized_target)]
            ic=None
            if len(finite)>=3 and finite.realized_target.nunique()>1 and finite[name].nunique()>1:
                # Average ranks handle ties; no scipy small-sample warnings.
                ic=float(finite.realized_target.rank().corr(finite[name].rank()))
                ics.append(ic)
            monthly.append({'model':name,'target_month':month,**metrics,'rank_ic':ic})
        summaries.append({'model':name,**result,'mean_monthly_rank_ic':float(np.mean(ics)) if ics else None,
                          'rank_ic_months':len(ics),'total_months':int(table.target_month.nunique())})
    return summaries,monthly


def label_scale(y,delta=1.):
    values=np.asarray(y,dtype=float)
    finite=values[np.isfinite(values)]
    if not len(finite): raise ValueError('no finite labels')
    return {'unit':'decimal return; 0.01 means 1%', 'n_total':len(values),'n_finite':len(finite),
            'quantiles':{str(q):float(np.quantile(finite,q)) for q in [0,.01,.5,.99,1]},
            'absolute_label_over_huber_delta_fraction':float((np.abs(finite)>delta).mean()),
            'huber_delta':delta}
