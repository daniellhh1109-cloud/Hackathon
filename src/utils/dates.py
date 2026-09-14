"""Calendar conventions. Cutoffs are exclusive UTC target-month starts."""
from datetime import date, datetime, timedelta, timezone
from calendar import monthrange


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def month_start(value):
    return as_date(value).replace(day=1)


def add_months(value, months):
    value = month_start(value)
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return date(year, month + 1, 1)


def month_end(value):
    value = month_start(value)
    return value.replace(day=monthrange(value.year, value.month)[1])


def target_month(eom):
    if as_date(eom) != month_end(eom):
        raise ValueError('eom must be a calendar month end')
    return add_months(eom, 1)


def decision_cutoff(target):
    if as_date(target) != month_start(target):
        raise ValueError('target_month must be a month start')
    return datetime.combine(as_date(target), datetime.min.time(), timezone.utc)


def filing_available_at(filing_date, delay_days=3):
    """Provider DATE + 3 calendar days at UTC midnight; an assumption, not SEC proof.

    Applies uniformly even if a vendor timestamp appears precise. A delay of at
    least one day prevents treating a date-only filing as available that morning.
    """
    if not isinstance(delay_days, int) or isinstance(delay_days, bool) or delay_days < 1:
        raise ValueError('delay_days must be an integer >= 1')
    return datetime.combine(as_date(filing_date), datetime.min.time(), timezone.utc) + timedelta(days=delay_days)
