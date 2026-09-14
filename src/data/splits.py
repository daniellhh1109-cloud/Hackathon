"""Member C: calendar alignment and guards; standard-library record API.

Records are mappings (DataFrames: use to_dict('records')). Prediction membership
never depends on label availability. Learned preprocessing must call the fit guard.
"""
from dataclasses import dataclass
from datetime import date
from src.utils.dates import (as_date, month_start, month_end, add_months,
                             target_month, decision_cutoff, filing_available_at)


@dataclass(frozen=True)
class AnnualSplit:
    year: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def partition(self, target):
        target = as_date(target)
        if target != month_start(target):
            raise ValueError('target_month must be a month start')
        for name in ('train', 'validation', 'test'):
            if getattr(self, name + '_start') <= target <= getattr(self, name + '_end'):
                return name
        return None


def annual_split(year):
    if year not in range(2021, 2027):
        raise ValueError('competition test years are 2021..2026')
    return AnnualSplit(year, date(2015, 2, 1), date(year-3, 12, 1),
                       date(year-2, 1, 1), date(year-1, 12, 1),
                       date(year, 1, 1), date(year, 8 if year == 2026 else 12, 1))


def split_records(records, year):
    split = annual_split(year)
    result = {key: [] for key in ('train', 'validation', 'test')}
    seen = set()
    for row in records:
        validate_alignment(row)
        key = (row['permno'], as_date(row['target_month']))
        if key in seen:
            raise ValueError('duplicate security / target_month')
        seen.add(key)
        name = split.partition(row['target_month'])
        if name:
            result[name].append(dict(row))
    return result


def validate_alignment(row):
    if row.get('permno') is None or str(row['permno']).strip() == '':
        raise ValueError('missing security identifier')
    if target_month(row['eom']) != as_date(row['target_month']):
        raise ValueError('target_month must equal eom plus one calendar month')
    if 'date' in row:
        source = as_date(row['date'])
        if month_start(source) != month_start(row['eom']) or source > as_date(row['eom']):
            raise ValueError('source date must belong to feature month')


def validate_quant_window(records, permno, target, length=12):
    decision_cutoff(target)
    if not isinstance(length, int) or length < 1:
        raise ValueError('length must be positive')
    rows = sorted(records, key=lambda r: as_date(r['eom']))
    expected = [month_end(add_months(target, n)) for n in range(-length, 0)]
    if len(rows) != length or [as_date(r['eom']) for r in rows] != expected:
        raise ValueError('quant window must contain exactly consecutive calendar months')
    for row in rows:
        validate_alignment(row)
        if row['permno'] != permno:
            raise ValueError('quant window contains another security')
    return rows


def select_filings(records, permno, target, delay_days=3):
    """Six filing-date calendar months, additionally subject to availability cutoff.

    Fail closed on unverified security mapping. No eligible text returns [], not
    a dropped numerical sample. Deduplication is within security/document only.
    """
    cutoff = decision_cutoff(target)
    lower = add_months(target, -6)
    selected, seen = [], set()
    for row in records:
        if row['permno'] != permno:
            continue
        filed = as_date(row['filing_date'])
        available = filing_available_at(filed, delay_days)
        if not (lower <= filed < as_date(target) and available < cutoff):
            continue
        if row.get('characteristics_link_basis') != 'characteristics_same_month':
            raise ValueError('filing needs independently verified same-month security mapping')
        key = row['document_id']
        if key in seen:
            raise ValueError('duplicate document for security')
        seen.add(key)
        selected.append({**row, 'available_at_utc': available.isoformat()})
    return sorted(selected, key=lambda r: (r['filing_date'], r['document_id']))


def validate_filings(records, permno, target, delay_days=3):
    rows = list(records)
    selected = select_filings(rows, permno, target, delay_days)
    if len(rows) != len(selected):
        raise ValueError('ineligible or future filing in sample')
    return selected


def validate_feature_columns(columns, whitelist):
    allowed = list(whitelist)
    if len(allowed) != 147 or len(set(allowed)) != 147:
        raise ValueError('expected verified 147-factor whitelist')
    forbidden = {'ret_exc_lead1m', 'ret', 'ret_exc', 'target_month', 'title_ai'}
    if forbidden.intersection(allowed):
        raise ValueError('contaminated whitelist')
    if len(columns) != len(set(columns)) or not set(columns) <= set(allowed):
        raise ValueError('model inputs contain duplicate or non-whitelisted fields')


def assert_fit_scope(records, year):
    """For scaler/imputer/feature selection fit, use train ONLY, never validation.

    Monthly labels are assumed realized at the next month start. This checks the
    annual boundary, not actual vendor delivery latency or historical revisions.
    """
    rows = list(records)
    if not rows:
        raise ValueError('empty fitting set')
    split = annual_split(year)
    for row in rows:
        validate_alignment(row)
        if split.partition(row['target_month']) != 'train':
            raise ValueError('preprocessor fit contains non-training rows')
        if add_months(row['target_month'], 1) > split.test_start:
            raise ValueError('label not realized before annual prediction')


def sample_label(window, permno, target):
    """Return the last FEATURE row's existing label verbatim, including missing."""
    return validate_quant_window(window, permno, target)[-1]['ret_exc_lead1m']

