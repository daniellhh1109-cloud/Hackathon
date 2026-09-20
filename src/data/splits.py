"""Shared numerical calendar/split guards. Filing selection is not implemented here."""
from dataclasses import dataclass
from datetime import date
from src.utils.dates import as_date, month_start, month_end, add_months, target_month


@dataclass(frozen=True)
class AnnualSplit:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def partition(self, month):
        month = as_date(month)
        if month != month_start(month):
            raise ValueError('target month must be month start')
        for role in ('train', 'validation', 'test'):
            if getattr(self, role + '_start') <= month <= getattr(self, role + '_end'):
                return role
        return None


def annual_split(year):
    if type(year) is not int or not 2021 <= year <= 2026:
        raise ValueError('year must be 2021..2026')
    return AnnualSplit(date(2015, 1, 1), date(year-3, 12, 1),
                       date(year-2, 1, 1), date(year-1, 12, 1),
                       date(year, 1, 1), date(year, 8 if year == 2026 else 12, 1))


def validate_alignment(row):
    eom, observed, target = map(as_date, (row['eom'], row['date'], row['target_month']))
    if eom != month_end(eom):
        raise ValueError('eom must be calendar month end')
    if month_start(observed) != month_start(eom) or observed > eom:
        raise ValueError('source date must be within feature month')
    if target != target_month(eom):
        raise ValueError('target month must be exactly the next calendar month')


def validate_feature_columns(columns, whitelist):
    columns, whitelist = list(columns), list(whitelist)
    forbidden = {'ret_exc_lead1m', 'ret', 'ret_exc', 'ret_exc_wins', 'target', 'target_month',
                 'permno', 'ticker', 'company_name', 'date', 'eom'}
    for values in (columns, whitelist):
        if len(values) != 147 or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError('expected 147 named factors')
        if len(set(values)) != 147 or forbidden.intersection(x.lower() for x in values):
            raise ValueError('duplicate or forbidden feature')
    if columns != whitelist:
        raise ValueError('feature order does not match whitelist')


def split_records(records, year):
    split, seen = annual_split(year), set()
    result = {name: [] for name in ('train', 'validation', 'test')}
    for row in records:
        validate_alignment(row)
        key = (row['permno'], as_date(row['target_month']))
        if key in seen:
            raise ValueError('duplicate security-target month')
        seen.add(key)
        role = split.partition(row['target_month'])
        if role:
            result[role].append(row)
    return result


def assert_fit_scope(records, year):
    split = annual_split(year)
    for row in records:
        validate_alignment(row)
        if split.partition(row['target_month']) != 'train':
            raise ValueError('fit records must belong to training period')


def validate_quant_window(records, permno, target):
    target = as_date(target)
    if target != month_start(target):
        raise ValueError('target must be month start')
    rows = sorted(records, key=lambda r: as_date(r['eom']))
    if len(rows) != 12:
        raise ValueError('expected 12 calendar months')
    for index, row in enumerate(rows):
        validate_alignment(row)
        if row['permno'] != permno or month_start(row['eom']) != add_months(target, index-12):
            raise ValueError('mixed securities or nonconsecutive calendar window')
    return rows


def sample_label(records, permno, target):
    return validate_quant_window(records, permno, target)[-1]['ret_exc_lead1m']
