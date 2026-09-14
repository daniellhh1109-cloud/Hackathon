"""Rebalance checks; weights are signed fractions of beginning NAV."""
from dataclasses import asdict, dataclass
import math
from src.utils.dates import as_date, month_start


def finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


@dataclass(frozen=True)
class ConstraintLimits:
    min_holdings: int = 100
    max_holdings: int = 500
    gross_limit: float = 2.0
    net_min: float = -0.5
    net_max: float = 0.5
    position_epsilon: float = 1e-8
    comparison_tolerance: float = 1e-10
    max_position: float | None = None  # Optional team limit, not official.
    beta_tolerance: float | None = None  # Input beta, not realized beta.

    def __post_init__(self):
        for name in ('min_holdings', 'max_holdings'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f'invalid {name}')
        if self.min_holdings > self.max_holdings:
            raise ValueError('invalid holdings range')
        for name in ('gross_limit', 'position_epsilon', 'comparison_tolerance',
                     'max_position', 'beta_tolerance'):
            v = getattr(self, name)
            if v is not None and finite(v, name) < 0:
                raise ValueError(f'invalid {name}')
        if finite(self.net_min, 'net_min') > finite(self.net_max, 'net_max'):
            raise ValueError('invalid net range')


def check_constraints(records, limits=None):
    """One month; required permno, target_month, weight; optional lagged beta_60m.

    Caller certifies beta provenance. Malformed inputs raise; failed limits are
    returned. Tiny weights are excluded ONLY from counts, never from exposures.
    """
    limits = limits or ConstraintLimits()
    rows = list(records)
    if not rows:
        raise ValueError('empty table: target month unknown')
    seen, months, weights, betas = set(), set(), [], []
    for row in rows:
        identifier = row.get('permno')
        if identifier is None or not str(identifier).strip():
            raise ValueError('missing permno')
        identifier = str(identifier).strip()
        if identifier in seen:
            raise ValueError('duplicate permno')
        seen.add(identifier)
        month = as_date(row['target_month'])
        if month != month_start(month):
            raise ValueError('target_month must be a month start')
        months.add(month)
        weights.append(finite(row['weight'], 'weight'))
        b = row.get('beta_60m')
        betas.append(None if b is None or b == '' else finite(b, 'beta_60m'))
    if len(months) != 1:
        raise ValueError('exactly one month required')
    active = [i for i, w in enumerate(weights) if abs(w) > limits.position_epsilon]
    gross, net = math.fsum(abs(w) for w in weights), math.fsum(weights)
    missing = sum(w != 0 and b is None for w, b in zip(weights, betas))
    beta = None if missing else math.fsum(w*b for w, b in zip(weights, betas) if w != 0)
    tol = limits.comparison_tolerance
    checks = {'holdings_count': limits.min_holdings <= len(active) <= limits.max_holdings,
              'gross': gross <= limits.gross_limit + tol,
              'net': limits.net_min - tol <= net <= limits.net_max + tol}
    team = {}
    if limits.max_position is not None:
        team['max_position'] = max(map(abs, weights)) <= limits.max_position + tol
    if limits.beta_tolerance is not None:
        team['input_beta'] = beta is not None and abs(beta) <= limits.beta_tolerance + tol
    return {'target_month': next(iter(months)).isoformat(), 'candidate_count': len(rows),
            'holdings_count': len(active), 'long_count': sum(weights[i] > 0 for i in active),
            'short_count': sum(weights[i] < 0 for i in active),
            'long_exposure': math.fsum(w for w in weights if w > 0),
            'short_exposure': math.fsum(-w for w in weights if w < 0),
            'gross': gross, 'net': net, 'max_abs_weight': max(map(abs, weights)),
            'input_beta': beta, 'missing_beta_count': missing,
            'official_numeric_checks': checks, 'team_checks': team,
            'official_numeric_pass': all(checks.values()),
            'configured_checks_pass': all(checks.values()) and all(team.values()),
            'limits': asdict(limits),
            'scope': 'Rebalance numeric checks only; not full compliance or realized beta certification.'}
