"""One-month linear-baseline samples using member C's calendar interfaces.

A linear baseline takes the latest feature row; a 12-month sequence is a separate
architecture choice and must use C.validate_quant_window when implemented.
"""
import numpy as np
import pandas as pd
from src.data.splits import split_records, validate_alignment
from src.utils.dates import target_month


KEYS = ['permno', 'date', 'eom', 'target_month']


def build_baseline_panel(raw):
    panel = raw.copy()
    ids = pd.to_numeric(panel['permno'], errors='raise')
    if ids.isna().any() or not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError('invalid permno')
    panel['permno'] = ids.astype('int64')
    for column in ('date', 'eom'):
        parsed = pd.to_datetime(panel[column], errors='raise')
        if parsed.isna().any():
            raise ValueError('missing date')
        panel[column] = parsed.dt.date
    targets = {eom: target_month(eom) for eom in panel['eom'].unique()}
    expected = panel['eom'].map(targets)
    if 'target_month' in panel and not pd.to_datetime(panel['target_month']).dt.date.equals(expected):
        raise ValueError('existing target_month conflicts with eom')
    panel['target_month'] = expected
    for row in panel[KEYS].to_dict('records'):
        validate_alignment(row)
    if panel.duplicated(['permno', 'eom']).any():
        raise ValueError('duplicate security-month')
    # Label remains on its original feature row; no shift or label-based filtering.
    panel['ret_exc_lead1m'] = pd.to_numeric(panel['ret_exc_lead1m'], errors='raise')
    return panel.sort_values(['eom', 'permno']).reset_index(drop=True)


def annual_indices(panel, year):
    records = panel[KEYS].copy()
    records['row_id'] = panel.index
    return {name: np.array([r['row_id'] for r in rows], dtype=int)
            for name, rows in split_records(records.to_dict('records'), year).items()}
