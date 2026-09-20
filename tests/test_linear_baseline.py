"""End-to-end leakage counterexamples and hand-calculated baseline checks."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from src.data.prepare_quant import monthly_rank_transform
from src.data.build_samples import build_baseline_panel, annual_indices
from src.models.linear_baseline import fit_year
from src.training.metrics import prediction_metrics

FACTORS = [f'factor_{i}' for i in range(147)]


def raw_panel():
    rows = []
    for month, offset in [('2018-11-30', 0), ('2018-12-31', 1), ('2020-12-31', 2), ('2021-01-31', 3)]:
        for security in range(1, 5):
            rows.append({'permno': security, 'date': month, 'eom': month,
                         'ret_exc_lead1m': (security - 2.5) * .02 + offset * .001,
                         **{f: float(security) for f in FACTORS}})
    return pd.DataFrame(rows)


def test_rank_hand_example_missing_ties_and_raw_preserved():
    panel = build_baseline_panel(raw_panel().iloc[:4])
    panel.loc[:, FACTORS[0]] = [1, np.nan, 3, 3]
    before = panel.copy(deep=True)
    x, _ = monthly_rank_transform(panel, FACTORS)
    # Median=3; filled [1,3,3,3], average ranks [1,3,3,3].
    np.testing.assert_allclose(x[FACTORS[0]], [-1, 1/3, 1/3, 1/3])
    pd.testing.assert_frame_equal(panel, before)


def test_all_missing_constant_infinity_and_singleton():
    panel = build_baseline_panel(raw_panel().iloc[:4])
    panel.loc[:, FACTORS[0]] = np.nan
    panel.loc[:, FACTORS[1]] = [np.inf, -np.inf, 7, 7]
    x, info = monthly_rank_transform(panel, FACTORS)
    assert (x[FACTORS[:2]] == 0).all().all()
    assert info['all_missing_month_factor_pairs'] == 1
    assert info['infinite_values_treated_as_missing'] == 2
    one, _ = monthly_rank_transform(panel.iloc[:1], FACTORS)
    assert (one == 0).all().all()


def test_future_month_changes_do_not_change_historical_ranks():
    panel = build_baseline_panel(raw_panel())
    before, _ = monthly_rank_transform(panel, FACTORS)
    panel.loc[panel.eom >= date(2020, 12, 31), FACTORS] = -999
    after, _ = monthly_rank_transform(panel, FACTORS)
    pd.testing.assert_frame_equal(before.iloc[:8], after.iloc[:8])


def test_source_and_label_preserved_and_c_split_used():
    raw = raw_panel()
    raw.loc[0, 'date'] = '2018-11-29'
    panel = build_baseline_panel(raw)
    assert panel.loc[0, 'date'] == date(2018, 11, 29)
    assert panel.loc[0, 'target_month'] == date(2018, 12, 1)
    assert panel.loc[0, 'ret_exc_lead1m'] == raw.loc[0, 'ret_exc_lead1m']
    indices = annual_indices(panel, 2021)
    assert indices['train'].tolist() == [0, 1, 2, 3]
    assert indices['validation'].tolist() == [4, 5, 6, 7]
    assert indices['test'].tolist() == list(range(8, 16))


@pytest.mark.parametrize('fault', ['duplicate', 'missing_id', 'bad_id', 'wrong_eom', 'future_source', 'wrong_target'])
def test_bad_samples_fail(fault):
    raw = raw_panel()
    if fault == 'duplicate': raw = pd.concat([raw, raw.iloc[:1]])
    if fault == 'missing_id': raw.loc[0, 'permno'] = np.nan
    if fault == 'bad_id': raw['permno'] = raw.permno.astype(float); raw.loc[0, 'permno'] = 1.5
    if fault == 'wrong_eom': raw.loc[0, 'eom'] = '2018-11-29'
    if fault == 'future_source': raw.loc[0, 'date'] = '2018-12-01'
    if fault == 'wrong_target': raw['target_month'] = '2021-01-01'
    with pytest.raises(ValueError): build_baseline_panel(raw)


def test_metric_zero_benchmark_and_undefined_denominator():
    result = prediction_metrics([1, -1, np.nan], [.5, -.5, 8])
    assert result['oos_r2_zero'] == .75
    assert result['n_predictions'] == 3 and result['n_scored'] == 2
    assert prediction_metrics([0], [1])['oos_r2_zero'] is None
    assert prediction_metrics([np.nan], [1])['mse'] is None
    assert prediction_metrics([1, 1], [0, 0])['oos_r2_zero'] == 0
    with pytest.raises(ValueError): prediction_metrics([1], [np.inf])


def test_test_labels_cannot_affect_model_or_membership():
    panel = build_baseline_panel(raw_panel())
    x, _ = monthly_rank_transform(panel, FACTORS)
    before, details, model = fit_year(panel, x, FACTORS, 2021, [1, 10])
    panel.loc[8:, 'ret_exc_lead1m'] = np.nan
    after, _, model2 = fit_year(panel, x, FACTORS, 2021, [1, 10])
    np.testing.assert_array_equal(before.prediction, after.prediction)
    np.testing.assert_array_equal(model.coef_, model2.coef_)
    assert len(after) == 8 and not after.label_available.any()
    assert details['partitions']['train']['last_target'] == '2018-12-01'


def test_validation_labels_do_not_enter_coefficient_fit():
    panel = build_baseline_panel(raw_panel())
    x, _ = monthly_rank_transform(panel, FACTORS)
    _, _, model = fit_year(panel, x, FACTORS, 2021, [10])
    panel.loc[4:7, 'ret_exc_lead1m'] = 100
    _, _, after = fit_year(panel, x, FACTORS, 2021, [10])
    np.testing.assert_array_equal(model.coef_, after.coef_)
    assert model.intercept_ == after.intercept_


@pytest.mark.parametrize('fault', ['label_feature', 'reordered', 'infinity', 'bad_alpha', 'no_train_labels'])
def test_model_rejects_bad_inputs(fault):
    panel = build_baseline_panel(raw_panel())
    x, _ = monthly_rank_transform(panel, FACTORS)
    alphas = [1]
    if fault == 'label_feature': x['ret_exc_lead1m'] = 0
    if fault == 'reordered': x = x[FACTORS[::-1]]
    if fault == 'infinity': x.iloc[0, 0] = np.inf
    if fault == 'bad_alpha': alphas = [0]
    if fault == 'no_train_labels': panel.loc[:3, 'ret_exc_lead1m'] = np.nan
    with pytest.raises(ValueError): fit_year(panel, x, FACTORS, 2021, alphas)
