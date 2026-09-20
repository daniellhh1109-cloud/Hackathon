"""Quant window, information boundary, full-universe ranking and trainer integration."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.quant_dataset import (normalize_panel, window_manifest, prepare_store,
                                    QuantDataset, make_loader)
from src.data.prepare_quant import monthly_rank_transform
from src.data.splits import annual_split, validate_quant_window
from src.training.trainer import annual_bounds, TrainingConfig, fit
from scripts.run_member_a import SmokeRegressor

FACTORS = [f'factor_{i}' for i in range(147)]


def raw_panel():
    rows = []
    for security in [1, 2, 3]:
        for idx, period in enumerate(pd.period_range('2017-01', '2021-02', freq='M')):
            eom = period.to_timestamp('M')
            rows.append({'permno': security, 'date': eom, 'eom': eom,
                         'ret_exc_lead1m': (idx+1)/1000, 'dolvol': security*1000,
                         **{f: float(security) for f in FACTORS}})
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    raw, factors = tmp_path/'raw.parquet', tmp_path/'factors.csv'
    raw_panel().to_parquet(raw)
    pd.DataFrame({'variable': FACTORS}).to_csv(factors, index=False)
    directory = tmp_path/'store'
    prepare_store(raw, factors, directory)
    return directory


def test_exact_window_original_label_and_batch(store):
    ds = QuantDataset(store, year=2021, partition='train')
    item = ds[0]
    assert item['quant'].shape == (12,147)
    assert item['target_month'] == '2018-01'
    assert item['quant_end_month'] == '2017-12'
    assert item['target'].item() == pytest.approx(.012)  # original last feature row, no shift
    timeline = ds.timeline(0)
    assert timeline.eom.dt.strftime('%Y-%m').tolist() == list(pd.period_range('2017-01','2017-12',freq='M').astype(str))
    assert timeline.permno.nunique() == 1
    assert make_loader(ds, batch_size=5).drop_last is False
    assert sum(len(b['quant']) for b in make_loader(ds, batch_size=5)) == len(ds)
    assert next(iter(make_loader(ds, batch_size=1)))['quant'].shape == (1,12,147)


def test_calendar_gap_and_listing_history():
    raw = raw_panel()
    raw = raw[~((raw.permno==1) & (raw.eom == pd.Timestamp('2017-06-30')))]
    panel = normalize_panel(raw)
    manifest = window_manifest(panel)
    assert manifest.iloc[0].window_status == 'insufficient_history'
    lookup = manifest.set_index(['permno','target_month'])
    assert lookup.loc[(1,'2018-02'),'window_status'] == 'calendar_gap'
    assert lookup.loc[(1,'2018-07'),'window_status'] == 'eligible'
    assert lookup.loc[(2,'2018-01'),'window_status'] == 'eligible'


def test_test_dataset_does_not_read_or_filter_labels(store):
    before = QuantDataset(store, year=2021, partition='test')
    samples = before.samples.copy()
    # Removing labels entirely must not prevent prediction dataset construction.
    (store/'labels.npy').unlink()
    after = QuantDataset(store, year=2021, partition='test')
    pd.testing.assert_frame_equal(samples, after.samples)
    assert 'target' not in after[0]
    torch.testing.assert_close(before[0]['quant'], after[0]['quant'])
    with pytest.raises(ValueError, match='membership'):
        QuantDataset(store, year=2021, partition='test', supervised=True)


def test_missing_training_labels_counted_without_removing_history(store):
    labels = np.load(store/'labels.npy')
    base = QuantDataset(store, year=2021, partition='train')
    row = base.rows[0]
    labels[row] = np.nan
    np.save(store/'labels.npy', labels)
    after = QuantDataset(store, year=2021, partition='train')
    assert len(after) == len(base)-1
    assert after.audit['excluded_missing_labels'] == 1
    assert after[0]['target_month'] == '2018-02'
    assert after[0]['quant'].shape == (12,147)  # missing earlier label does not remove feature history


def test_full_month_preprocessing_future_invariance_and_raw_context(store):
    panel = normalize_panel(raw_panel())
    before_raw = panel.copy(deep=True)
    first, _ = monthly_rank_transform(panel, FACTORS)
    pd.testing.assert_frame_equal(panel, before_raw)
    panel.loc[panel.eom >= '2021-01-01', FACTORS] = -999
    second, _ = monthly_rank_transform(panel, FACTORS)
    pd.testing.assert_frame_equal(first.loc[panel.eom < '2021-01-01'], second.loc[panel.eom < '2021-01-01'])
    context = pd.read_parquet(store/'raw_context.parquet')
    assert set(context.dolvol) == {1000,2000,3000}
    assert 'ret_exc_lead1m' not in context
    ds = QuantDataset(store, year=2021, partition='train')
    assert (ds[0]['quant'] == -1).all()  # ranks from all 3 securities


@pytest.mark.parametrize('fault', ['duplicate','eom','date','target','id'])
def test_invalid_raw_rejected(fault):
    raw = raw_panel()
    if fault == 'duplicate': raw = pd.concat([raw, raw.iloc[:1]])
    elif fault == 'eom': raw.loc[0,'eom'] = pd.Timestamp('2017-01-30')
    elif fault == 'date': raw.loc[0,'date'] = pd.Timestamp('2017-02-01')
    elif fault == 'target': raw['target_month'] = '2020-01-01'
    else: raw.loc[0,'permno'] = -1
    with pytest.raises(ValueError): normalize_panel(raw)


def test_a_trainer_accepts_b_datasets(store, tmp_path):
    train = QuantDataset(store, year=2021, partition='train')
    validation = QuantDataset(store, year=2021, partition='validation')
    model, summary = fit(SmokeRegressor, train, validation,
        config=TrainingConfig(max_epochs=2, batch_size=16), target_year=2021,
        output_dir=tmp_path/'fit', feature_names=train.feature_names,
        model_metadata={'name':'SmokeRegressor'}, preprocessing_metadata=train.metadata['preprocessing'])
    assert summary['epochs_run'] == 2 and not model.training


@pytest.mark.parametrize('year', range(2021,2027))
def test_shared_boundaries_match_a(year):
    expected = annual_bounds(year)
    split = annual_split(year)
    for part in ['train','validation','test']:
        assert expected[part] == [getattr(split,f'{part}_{edge}').strftime('%Y-%m') for edge in ['start','end']]


def test_explicit_window_guard():
    panel = normalize_panel(raw_panel()).iloc[:12]
    rows = panel.to_dict('records')
    for row in rows: row['target_month'] = row['target_month'] + '-01'
    assert len(validate_quant_window(rows, 1, '2018-01-01')) == 12
    rows[0]['permno'] = 2
    with pytest.raises(ValueError): validate_quant_window(rows, 1, '2018-01-01')


@pytest.mark.parametrize('fault', ['row_index','mixed_security','eligibility','target_month','continuous_months'])
def test_corrupt_window_cache_rejected_on_load(store, fault):
    path=store/'windows.parquet'
    windows=pd.read_parquet(path)
    index=windows.index[windows.window_status.eq('eligible')][0]
    if fault=='row_index': windows.loc[index,'row_index']=0
    elif fault=='mixed_security': windows.loc[index,'permno']=999999
    elif fault=='eligibility': windows.loc[0,'window_status']='eligible'
    elif fault=='target_month': windows.loc[index,'target_month']='2021-01'
    else: windows.loc[index,'continuous_months']=999
    windows.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='window index differs'):
        QuantDataset(store,year=2021,partition='train')


def test_unsorted_context_cache_rejected(store):
    path=store/'raw_context.parquet'
    pd.read_parquet(path).iloc[::-1].to_parquet(path,index=False)
    with pytest.raises(ValueError,match='sorted'):
        QuantDataset(store,year=2021,partition='test')


@pytest.mark.parametrize('row', [-1, *range(11), 150, 151])
def test_window_access_rejects_invalid_end_rows(store, row):
    ds = QuantDataset(store, year=2021, partition='train')
    assert len(ds.quant) == 150
    # Simulate an in-memory index mutation after constructor validation.
    ds.rows = ds.rows.copy()
    ds.rows[0] = row
    with pytest.raises(ValueError, match='invalid 12-month window end row'):
        ds[0]
    with pytest.raises(ValueError, match='invalid 12-month window end row'):
        ds.timeline(0)


@pytest.mark.parametrize('edge', ['first', 'last'])
def test_window_access_accepts_array_boundaries(store, edge):
    ds = QuantDataset(store, year=2021, partition='test', supervised=False)
    row = 11 if edge == 'first' else len(ds.quant) - 1
    ds.rows = ds.rows.copy()
    ds.rows[0] = row
    torch.testing.assert_close(
        ds[0]['quant'], torch.from_numpy(ds.quant[row-11:row+1].copy())
    )
    assert ds[0]['quant'].shape == (12, 147)
    assert len(ds.timeline(0)) == 12
