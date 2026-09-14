from datetime import date
import pytest
from src.utils.dates import add_months, month_end
from src.data.splits import (annual_split, split_records, select_filings,
    validate_filings, validate_feature_columns, assert_fit_scope)


def filing(day='2020-12-28', **kw):
    return dict(permno=12490, document_id='a', filing_date=day,
                characteristics_link_basis='characteristics_same_month', **kw)


def row(target, label=None):
    return dict(permno=12490, eom=month_end(add_months(target, -1)),
                target_month=target, ret_exc_lead1m=label)


@pytest.mark.parametrize('year', range(2021, 2027))
def test_all_annual_boundaries(year):
    s = annual_split(year)
    assert s.train_start == date(2015, 2, 1)
    assert s.train_end == date(year-3, 12, 1)
    assert s.validation_start == date(year-2, 1, 1)
    assert s.validation_end == date(year-1, 12, 1)
    assert s.test_end == date(year, 8 if year == 2026 else 12, 1)
    for name in ('train', 'validation', 'test'):
        assert s.partition(getattr(s, name+'_start')) == name
        assert s.partition(getattr(s, name+'_end')) == name
    assert s.partition(add_months(s.train_end, 1)) == 'validation'
    assert s.partition(add_months(s.validation_end, 1)) == 'test'
    assert s.partition(add_months(s.test_end, 1)) is None
    assert s.partition('2015-01-01') is None


def test_68_months():
    months = []
    for year in range(2021, 2027):
        s = annual_split(year)
        months += [add_months(s.test_start, i) for i in range(s.test_end.month)]
    assert len(months) == len(set(months)) == 68
    assert months == [add_months('2021-01-01', i) for i in range(68)]


@pytest.mark.parametrize('day', ['2020-12-29', '2020-12-31', '2021-01-01', '2020-06-30'])
def test_future_and_outside_window_rejected(day):
    f = filing(day, content_report_date='2020-01-01')
    assert select_filings([f], 12490, '2021-01-01') == []
    with pytest.raises(ValueError): validate_filings([f], 12490, '2021-01-01')


def test_text_lower_bound_empty_and_other_security():
    assert len(select_filings([filing('2020-07-01')], 12490, '2021-01-01')) == 1
    assert len(select_filings([filing()], 12490, '2021-01-01')) == 1
    assert select_filings([], 12490, '2021-01-01') == []
    with pytest.raises(ValueError): validate_filings([filing()], 7, '2021-01-01')


def test_duplicate_and_unverified_mapping():
    with pytest.raises(ValueError): select_filings([filing(), filing()], 12490, '2021-01-01')
    f = filing(); f['characteristics_link_basis'] = 'historical_gap'
    with pytest.raises(ValueError): select_filings([f], 12490, '2021-01-01')


def test_label_missingness_does_not_filter_prediction():
    records = [row('2021-01-01'), row('2021-02-01', 0.5)]
    assert len(split_records(records, 2021)['test']) == 2
    assert records[1]['ret_exc_lead1m'] == 0.5
    with pytest.raises(ValueError): split_records(records*2, 2021)


@pytest.mark.parametrize('target', ['2019-01-01', '2020-12-01', '2021-01-01', '2026-09-01'])
def test_preprocessing_rejects_nontrain(target):
    with pytest.raises(ValueError): assert_fit_scope([row(target)], 2021)


def test_preprocessing_accepts_train():
    assert_fit_scope([row('2015-02-01'), row('2018-12-01')], 2021)


@pytest.mark.parametrize('field', ['ret_exc_lead1m', 'ret', 'target_month', 'title_ai', 'provider_added_at_utc'])
def test_future_fields_not_model_inputs(field):
    whitelist = [f'factor_{i}' for i in range(147)]
    validate_feature_columns(whitelist, whitelist)
    with pytest.raises(ValueError): validate_feature_columns(whitelist + [field], whitelist)
