from datetime import date
import pytest
from src.utils.dates import add_months, month_end, target_month
from src.data.splits import validate_quant_window, sample_label, validate_alignment


def window():
    return [dict(permno=12490, eom=month_end(add_months('2021-01-01', i)),
                 target_month=add_months('2021-01-01', i+1), ret_exc_lead1m=i)
            for i in range(-12, 0)]


def test_leap_year_and_year_rollover():
    assert month_end('2020-02-01') == date(2020, 2, 29)
    assert target_month('2020-12-31') == date(2021, 1, 1)
    assert target_month('2026-07-31') == date(2026, 8, 1)


def test_unsorted_window_and_label_not_shifted():
    rows = window()[::-1]
    assert len(validate_quant_window(rows, 12490, '2021-01-01')) == 12
    assert sample_label(rows, 12490, '2021-01-01') == -1
    rows[0]['ret_exc_lead1m'] = None
    assert sample_label(rows, 12490, '2021-01-01') is None


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'security', 'target', 'eom', 'source'])
def test_bad_windows_rejected(fault):
    rows = window()
    if fault == 'missing': rows.pop(4)
    if fault == 'duplicate': rows[4] = rows[3].copy()
    if fault == 'security': rows[4]['permno'] = 1
    if fault == 'target': rows[-1]['target_month'] = '2021-02-01'
    if fault == 'eom': rows[1]['eom'] = '2020-02-28'
    if fault == 'source': rows[-1]['date'] = '2021-01-01'
    with pytest.raises(ValueError): validate_quant_window(rows, 12490, '2021-01-01')


def test_source_date_preserved():
    row = dict(permno=1, date='2020-02-28', eom='2020-02-29', target_month='2020-03-01')
    validate_alignment(row)
    assert row['date'] == '2020-02-28'
