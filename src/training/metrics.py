"""Prediction metrics against a ZERO excess-return forecast (not mean-centered R²)."""
import numpy as np


def prediction_metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape:
        raise ValueError('expected equal one-dimensional arrays')
    if not np.isfinite(prediction).all():
        raise ValueError('non-finite predictions')
    usable = np.isfinite(y)
    actual, estimated = y[usable], prediction[usable]
    sse = float(np.sum((actual - estimated) ** 2))
    zero_sse = float(np.sum(actual ** 2))
    return {'n_predictions': len(y), 'n_scored': int(usable.sum()),
            'n_missing_or_nonfinite_labels': int((~usable).sum()),
            'sse': sse, 'zero_prediction_sse': zero_sse,
            'mse': sse / len(actual) if len(actual) else None,
            'oos_r2_zero': 1 - sse / zero_sse if zero_sse > 0 else None}
