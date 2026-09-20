"""Calendar dates used by numerical samples and portfolio checks."""
from calendar import monthrange
from datetime import date, datetime


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid date: {value!r}') from exc


def month_start(value):
    return as_date(value).replace(day=1)


def month_end(value):
    value = as_date(value)
    return value.replace(day=monthrange(value.year, value.month)[1])


def add_months(value, count):
    value = month_start(value)
    year, month = divmod(value.year * 12 + value.month - 1 + count, 12)
    return date(year, month + 1, 1)


def target_month(value):
    return add_months(value, 1)
