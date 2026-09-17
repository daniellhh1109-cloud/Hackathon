"""Artificial fixtures only, never registered as a production optimizer/backtester."""
from datetime import timedelta
from src.utils.dates import as_date


def components(month='2021-01-01', initial=True):
    cutoff = (as_date(month) - timedelta(days=1)).isoformat()
    predictions, context = [], []
    for i in range(320):
        predictions.append({'permno': str(10000+i), 'target_month': month,
            'predicted_excess_return': (160-i)/10000, 'information_cutoff': cutoff,
            'model_selection_end': '2020-12-31', 'checkpoint_id': 'SYNTHETIC-NOT-A-TRAINED-MODEL'})
        context.append({'permno': str(10000+i), 'target_month': month, 'available_at': cutoff,
                        'beta_60m': 1.0, 'market_cap': 1e9, 'dollar_volume': 1e7})
    return predictions, context, {'target_month': month, 'as_of': cutoff,
        'kind': 'initial' if initial else 'drifted', 'weights': []}


def equal_weight_fixture(*, candidates, previous_weights, policy):
    """Deterministic test stub, NOT member C's optimized portfolio."""
    counts = {side: sum(r['side'] == side for r in candidates) for side in ('long', 'short')}
    return {'status': 'optimal', 'weights': [{'permno': r['permno'],
        'weight': (1 if r['side'] == 'long' else -1)/counts[r['side']]} for r in candidates]}


def evaluate_fixture(*, frozen_holdings_path, output_dir):
    """Artificial post-commit evaluation probe, NOT a financial backtest."""
    import json
    from pathlib import Path
    rows = json.loads(Path(frozen_holdings_path).read_text())
    return {'synthetic': True, 'purpose': 'verify post-commit evaluation ordering only',
            'positions_read': len(rows)}
