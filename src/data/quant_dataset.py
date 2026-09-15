"""Disk-backed continuous 12-month quant windows; no pre-expanded window tensor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.prepare_quant import load_factors, monthly_rank_transform
from src.data.splits import annual_split, validate_feature_columns

TARGET = 'ret_exc_lead1m'
CONTEXT = ['beta_60m', 'me', 'market_equity', 'dolvol', 'prc', 'ticker', 'company_name',
           'ticker_name_reference_date', 'ticker_name_source', 'ticker_name_status', 'gics', 'sic']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def normalize_panel(raw):
    """Validate and sort by security/month, without any label-based row filtering."""
    panel = raw.copy()
    ids = pd.to_numeric(panel['permno'], errors='raise')
    if ids.isna().any() or not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError('invalid permno')
    panel['permno'] = ids.astype('int64')
    for name in ['date', 'eom']:
        parsed = pd.to_datetime(panel[name], errors='raise')
        if parsed.isna().any() or not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError('missing/non-date source date')
        panel[name] = parsed
    if not panel.eom.dt.is_month_end.all():
        raise ValueError('eom must be calendar month end')
    if not panel.date.dt.to_period('M').equals(panel.eom.dt.to_period('M')):
        raise ValueError('source date must be in feature month')
    expected = (panel.eom.dt.to_period('M') + 1).astype(str)
    if 'target_month' in panel:
        observed = pd.to_datetime(panel.target_month, errors='raise')
        if not observed.dt.is_month_start.all() or not observed.dt.to_period('M').astype(str).equals(expected):
            raise ValueError('existing target_month mismatch')
    panel['target_month'] = expected
    if panel.duplicated(['permno', 'eom']).any():
        raise ValueError('duplicate security-month')
    if TARGET in panel:
        panel[TARGET] = pd.to_numeric(panel[TARGET], errors='raise')
    return panel.sort_values(['permno', 'eom']).reset_index(drop=True)


def window_manifest(panel):
    """Every feature row gets an explicit eligibility result; gaps restart history."""
    ids = panel.permno.to_numpy()
    months = panel.eom.dt.year.to_numpy() * 12 + panel.eom.dt.month.to_numpy() - 1
    continuation = np.zeros(len(panel), dtype=bool)
    continuation[1:] = (ids[1:] == ids[:-1]) & (months[1:] == months[:-1] + 1)
    starts = np.maximum.accumulate(np.where(~continuation, np.arange(len(panel)), 0))
    history = np.arange(len(panel)) - starts + 1
    # Separate short listing history from a gap following earlier observations.
    security_start = np.maximum.accumulate(np.where(np.r_[True, ids[1:] != ids[:-1]], np.arange(len(panel)), 0))
    enough_prior_rows = np.arange(len(panel)) - security_start + 1 >= 12
    reason = np.where(history >= 12, 'eligible', np.where(enough_prior_rows, 'calendar_gap', 'insufficient_history'))
    return pd.DataFrame({'row_index': np.arange(len(panel)), 'permno': ids,
                         'target_month': panel.target_month,
                         'quant_end_month': panel.eom.dt.strftime('%Y-%m'),
                         'continuous_months': history, 'window_status': reason})


def validate_window_index(windows, context, n_rows):
    """Recompute calendar eligibility from source identities, not cached claims.

    This verifies row alignment without opening labels or materializing windows.
    It cannot certify that the upstream provider's factors were point-in-time.
    """
    if len(windows) != n_rows or len(context) != n_rows or n_rows == 0:
        raise ValueError('window/context row-count mismatch')
    normalized = normalize_panel(context)
    # Never silently reorder context: quant.npy uses the original row positions.
    original_keys = context[['permno', 'eom']].copy()
    original_keys['eom'] = pd.to_datetime(original_keys['eom'])
    if not np.array_equal(original_keys.permno.to_numpy(), normalized.permno.to_numpy()) or not np.array_equal(
            original_keys.eom.to_numpy(), normalized.eom.to_numpy()):
        raise ValueError('context rows must be sorted by security and month')
    expected = window_manifest(normalized)
    if not set(expected.columns) <= set(windows.columns):
        raise ValueError('window index missing required columns')
    # IDs and offsets must be integers, not float indices that get truncated later.
    for column in ['row_index', 'permno', 'continuous_months']:
        if not pd.api.types.is_integer_dtype(windows[column].dtype):
            raise ValueError(f'window {column} must be integer')
    try:
        pd.testing.assert_frame_equal(windows[expected.columns].reset_index(drop=True), expected,
                                      check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('window index differs from actual calendar/identity records') from exc


def prepare_store(raw_path, factor_path, output_dir):
    """Prepare all months/securities; refuse to overwrite, write manifest last."""
    import pyarrow.parquet as pq
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    factors = load_factors(factor_path)
    schema = pq.ParquetFile(raw_path).schema.names
    required = ['permno', 'date', 'eom', TARGET] + factors
    missing = sorted(set(required) - set(schema))
    if missing:
        raise ValueError(f'missing columns: {missing}')
    columns = list(dict.fromkeys(required + [c for c in CONTEXT + ['target_month'] if c in schema]))
    print('Reading full monthly universe...', flush=True)
    panel = normalize_panel(pd.read_parquet(raw_path, columns=columns))
    if panel.empty:
        raise ValueError('empty input panel')
    print(f'Preprocessing {len(panel):,} rows x {len(factors)} factors...', flush=True)
    features, diagnostics = monthly_rank_transform(panel, factors)
    manifest = window_manifest(panel)
    labels = panel[TARGET].to_numpy(dtype=np.float64)
    monthly = manifest.groupby(['target_month', 'window_status']).size().unstack(fill_value=0)
    for status in ['eligible', 'calendar_gap', 'insufficient_history']:
        if status not in monthly:
            monthly[status] = 0
    eligible = manifest.window_status.eq('eligible').to_numpy()
    monthly['eligible_missing_label'] = manifest.loc[eligible & ~np.isfinite(labels)].groupby('target_month').size().reindex(monthly.index, fill_value=0)
    monthly['eligible_supervised'] = monthly['eligible'] - monthly['eligible_missing_label']
    audit = {'rows': len(panel), 'securities': int(panel.permno.nunique()),
             'feature_month_range': [panel.eom.min().strftime('%Y-%m'), panel.eom.max().strftime('%Y-%m')],
             'window_counts': {str(k): int(v) for k, v in manifest.window_status.value_counts().items()},
             'eligible_missing_label': int((eligible & ~np.isfinite(labels)).sum()),
             'preprocessing': diagnostics, 'exclusion_precedence': 'window eligibility first, then finite labels for supervision only'}
    output_dir.mkdir(parents=True)
    np.save(output_dir / 'quant.npy', features.to_numpy(dtype=np.float32, copy=False), allow_pickle=False)
    # Keep labels physically separate: inference constructor does not read them.
    np.save(output_dir / 'labels.npy', labels, allow_pickle=False)
    manifest.to_parquet(output_dir / 'windows.parquet', index=False)
    context_columns = list(dict.fromkeys(['permno', 'date', 'eom', 'target_month'] + [c for c in CONTEXT if c in panel]))
    panel[context_columns].to_parquet(output_dir / 'raw_context.parquet', index=False)
    monthly.to_csv(output_dir / 'monthly_audit.csv')
    write_json(output_dir / 'audit.json', audit)
    metadata = {'format_version': 1, 'n_rows': len(panel), 'feature_names': factors,
                'source': {'path': str(Path(raw_path).resolve()), 'sha256': sha256(raw_path)},
                'factor_list': {'path': str(Path(factor_path).resolve()), 'sha256': sha256(factor_path)},
                'preprocessing': {'name': 'same_month_median_average_rank_v1', 'rank_range': [-1, 1],
                                  'all_missing_policy': 'zero', 'infinity_policy': 'missing',
                                  'universe': 'all source securities per month before window/label filtering',
                                  'dtype': 'float32', 'return_unit': 'decimal', 'quant_window': 12},
                'context_warning': 'Raw ticker/name reference dates may be posterior; display-only unless point-in-time verified.'}
    write_json(output_dir / 'metadata.json', metadata)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return audit


class QuantDataset(Dataset):
    """Training items match A; inference mode never reads or returns realized labels."""
    def __init__(self, store_dir, *, year, partition, supervised=None):
        if partition not in ('train', 'validation', 'test'):
            raise ValueError('partition must be train, validation or test')
        if supervised is None:
            supervised = partition != 'test'
        if partition == 'test' and supervised:
            raise ValueError('test membership must not depend on labels; score separately')
        self.store_dir = Path(store_dir)
        self.metadata = json.loads((self.store_dir / 'metadata.json').read_text())
        if self.metadata['format_version'] != 1:
            raise ValueError('unsupported store format')
        self.feature_names = self.metadata['feature_names']
        validate_feature_columns(self.feature_names, self.feature_names)
        self.quant = np.load(self.store_dir / 'quant.npy', mmap_mode='r', allow_pickle=False)
        if self.quant.shape != (self.metadata['n_rows'], 147) or self.quant.dtype != np.dtype('float32'):
            raise ValueError('quant array shape mismatch')
        windows = pd.read_parquet(self.store_dir / 'windows.parquet')
        context = pd.read_parquet(self.store_dir / 'raw_context.parquet',
                                  columns=['permno', 'date', 'eom', 'target_month'])
        validate_window_index(windows, context, self.metadata['n_rows'])
        split = annual_split(year)
        low, high = [getattr(split, f'{partition}_{edge}').strftime('%Y-%m') for edge in ('start','end')]
        in_period = windows.target_month.between(low, high)
        usable = in_period & windows.window_status.eq('eligible')
        self.labels = None
        missing = 0
        if supervised:
            self.labels = np.load(self.store_dir / 'labels.npy', mmap_mode='r', allow_pickle=False)
            if self.labels.shape != (self.metadata['n_rows'],):
                raise ValueError('label array shape mismatch')
            finite = np.isfinite(self.labels[windows.row_index.to_numpy()])
            missing = int((usable & ~finite).sum())
            usable &= finite
        self.samples = windows.loc[usable].reset_index(drop=True)
        self.rows = self.samples.row_index.to_numpy()
        self.permnos = self.samples.permno.to_numpy()
        self.targets = self.samples.target_month.tolist()
        self.ends = self.samples.quant_end_month.tolist()
        self.audit = {'year': year, 'partition': partition, 'bounds': [low, high],
                      'input_rows_in_period': int(in_period.sum()),
                      'excluded_window': int((in_period & windows.window_status.ne('eligible')).sum()),
                      'excluded_missing_labels': missing if supervised else None,
                      'label_filter_applied': bool(supervised), 'samples': len(self.samples)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        result = {'quant': torch.from_numpy(self.quant[row-11:row+1].copy()),
                  'permno': int(self.permnos[index]), 'target_month': self.targets[index],
                  'quant_end_month': self.ends[index]}
        if self.labels is not None:
            result['target'] = torch.tensor(float(self.labels[row]), dtype=torch.float32)
        return result

    def timeline(self, index):
        row = int(self.rows[index])
        context = pd.read_parquet(self.store_dir / 'raw_context.parquet').iloc[row-11:row+1]
        result = context.copy()
        result['sample_target_month'] = self.targets[index]
        return result


def make_loader(dataset, *, batch_size=512, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)

