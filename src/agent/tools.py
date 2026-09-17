"""Point-in-time tool boundary and optimizer adapter for member A.

Adapters are trusted local code, not LLM-authored programs. No raw data files or
realized-return tables are passed through this API. Frozen policy is not editable
by a controller. Input provenance still requires member B's source audit.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
from typing import Protocol

from src.portfolio.risk import ConstraintLimits, check_constraints, finite
from src.utils.dates import as_date, month_start


class ToolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def require(condition, code, message):
    if not condition:
        raise ToolError(code, message)


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    n_long: int = 150
    n_short: int = 150
    # Filter before score ranking; never swap predictions for LLM stock picks.
    min_dollar_volume: float = 0.0
    max_abs_beta: float = 5.0

    def __post_init__(self):
        require(isinstance(self.name, str) and bool(self.name), 'CONFIG', 'plan name required')
        for value in (self.n_long, self.n_short):
            require(type(value) is int and value > 0, 'CONFIG', 'positive integer candidate counts required')
        for value in (self.min_dollar_volume, self.max_abs_beta):
            require(finite(value, 'plan limit') >= 0, 'CONFIG', 'negative plan limit')


@dataclass(frozen=True)
class AgentPolicy:
    plans: tuple = (CandidatePlan('default'), CandidatePlan('expanded', 200, 200))
    limits: ConstraintLimits = ConstraintLimits(max_position=.02, beta_tolerance=.10)
    net_target_tol: float = .05
    lambda_turnover: float = .01
    lambda_l2: float = .001
    min_position: float = .001
    gross_target: float = 2.0
    max_optimizer_retries: int = 3  # Initial call plus at most three retries.
    max_tool_calls: int = 24

    def __post_init__(self):
        require(bool(self.plans) and len({p.name for p in self.plans}) == len(self.plans),
                'CONFIG', 'unique nonempty candidate plans required')
        require(type(self.max_optimizer_retries) is int and 0 <= self.max_optimizer_retries <= 10,
                'CONFIG', 'optimizer retry limit must be 0..10')
        require(type(self.max_tool_calls) is int and 7 <= self.max_tool_calls <= 100,
                'CONFIG', 'tool limit must be 7..100')
        require(self.limits.position_epsilon < finite(self.min_position, 'min_position')
                and 0 < finite(self.gross_target, 'gross_target') <= self.limits.gross_limit,
                'CONFIG', 'invalid minimum position or gross target')
        for value in (self.net_target_tol, self.lambda_turnover, self.lambda_l2):
            require(finite(value, 'policy value') >= 0, 'CONFIG', 'negative policy value')
        # Even trusted config cannot quietly relax official rebalance limits.
        require(100 <= self.limits.min_holdings <= self.limits.max_holdings <= 500
                and self.limits.gross_limit <= 2 and self.limits.net_min >= -.5
                and self.limits.net_max <= .5
                and self.limits.comparison_tolerance <= 1e-6
                and self.limits.position_epsilon <= 1e-6, 'CONFIG', 'official limits cannot be relaxed')


class Optimizer(Protocol):
    """C supplies an implementation. Previous weights include ALL exiting names.

    Return {status: 'optimal'|'failed', weights: [{permno, weight}],
            failure_reason?: str}. A independently validates the weights.
    """
    def __call__(self, *, candidates: list, previous_weights: list, policy: dict) -> dict: ...


def _schema(name, description, properties=None):
    properties = properties or {}
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': list(properties), 'additionalProperties': False}}


def tool_schemas(policy):
    return [
        _schema('get_predictions', 'Get anonymous, point-in-time stock scores.'),
        _schema('get_company_context', 'Get anonymous beta and liquidity context.'),
        _schema('get_previous_holdings', 'Get prior available drifted weights, including exits.'),
        _schema('rank_candidates', 'Apply one approved score-based candidate plan.',
                {'plan': {'type': 'string', 'enum': [p.name for p in policy.plans]}}),
        _schema('optimize_weights', 'Call the registered deterministic optimizer; no weights accepted.'),
        _schema('check_constraints', 'Independently check the most recent optimizer result.'),
        _schema('write_portfolio_report', 'Finalize a checked portfolio and evidence-based report.'),
    ]


class PortfolioTools:
    """One instance per month. Construct only from vetted, point-in-time snapshots.

    predictions: target_month, permno, predicted_excess_return, information_cutoff,
                 checkpoint_id. context: target_month, permno, available_at,
                 beta_60m, market_cap, dollar_volume.
    previous: target_month (THIS decision month), as_of (previous calendar month
              end), kind='drifted', weights=[{permno, weight}]. Explicit kind='initial'
              is allowed only with empty weights by run orchestration.
    """
    def __init__(self, month, predictions, context, previous, optimizer, policy=None):
        self.month = as_date(month)
        require(self.month == month_start(self.month), 'DATE', 'month must be ISO month start')
        self.cutoff = self.month - timedelta(days=1)
        self.policy = policy or AgentPolicy()
        self.optimizer = optimizer
        self.reads = set()
        self.candidates = None
        self.weights = None
        self.checked = False
        self.report = None
        self.attempts = 0
        self.plan_name = None
        self.used_plans = []
        self.last_error = None
        self.check_report = None
        p = self._index(predictions, 'predictions')
        c = self._index(context, 'context')
        require(bool(p) and set(p) == set(c), 'DATA', 'prediction/context keys must match exactly')
        self.rows = {}
        for identifier in sorted(p):
            pred, ctx = p[identifier], c[identifier]
            require(as_date(pred['information_cutoff']) == self.cutoff, 'FUTURE_DATA',
                    'prediction cutoff must be previous month end')
            require(as_date(ctx['available_at']) <= self.cutoff, 'FUTURE_DATA', 'future context rejected')
            require(as_date(pred['model_selection_end']) <= self.cutoff, 'FUTURE_DATA',
                    'model selection cannot use future observations')
            require(isinstance(pred['checkpoint_id'], str) and bool(pred['checkpoint_id']),
                    'DATA', 'checkpoint_id required')
            self.rows[identifier] = dict(permno=identifier,
                target_month=self.month.isoformat(),
                predicted_excess_return=finite(pred['predicted_excess_return'], 'prediction'),
                beta_60m=finite(ctx['beta_60m'], 'beta'),
                market_cap=finite(ctx['market_cap'], 'market_cap'),
                dollar_volume=finite(ctx['dollar_volume'], 'dollar_volume'))
            require(self.rows[identifier]['market_cap'] > 0 and self.rows[identifier]['dollar_volume'] >= 0,
                    'DATA', 'invalid size/liquidity')
        self.previous = deepcopy(previous)
        require(as_date(previous['target_month']) == self.month and as_date(previous['as_of']) == self.cutoff,
                'FUTURE_DATA', 'prior state must be at previous month end for this decision')
        require(previous['kind'] in ('initial', 'drifted'), 'DATA', 'prior kind must be initial/drifted')
        require(previous['kind'] != 'initial' or not previous['weights'], 'DATA', 'initial state must be empty')
        self.previous_weights = []
        seen = set()
        for row in previous['weights']:
            identifier = self._id(row['permno'])
            require(identifier not in seen, 'DATA', 'duplicate previous position')
            seen.add(identifier)
            self.previous_weights.append({'permno': identifier, 'weight': finite(row['weight'], 'previous weight')})
        # Identity minimization: no ticker, company name, calendar dates, free text,
        # realized labels or checkpoint names enter the controller's observations.
        self.aliases = {k: f'asset_{i:05d}' for i, k in enumerate(sorted(set(p) | seen))}
        self.provenance = {'predictions_sha256': digest(predictions), 'context_sha256': digest(context),
                           'previous_sha256': digest(previous),
                           'checkpoint_ids': sorted({v['checkpoint_id'] for v in p.values()})}

    @staticmethod
    def _id(value):
        require(not isinstance(value, bool) and isinstance(value, (str, int)) and bool(str(value).strip()),
                'DATA', 'invalid permno')
        return str(value).strip()

    def _index(self, rows, name):
        output = {}
        for row in rows:
            require(as_date(row['target_month']) == self.month, 'DATE', f'{name}: wrong target month')
            identifier = self._id(row['permno'])
            require(identifier not in output, 'DATA', f'{name}: duplicate permno')
            output[identifier] = row
        return output

    def snapshot(self):
        return {'reads': sorted(self.reads), 'has_candidates': self.candidates is not None,
                'has_weights': self.weights is not None, 'checked': self.checked,
                'finished': self.report is not None, 'optimizer_attempts': self.attempts,
                'max_optimizer_attempts': self.policy.max_optimizer_retries + 1,
                'approved_plans': [asdict(p) for p in self.policy.plans],
                'used_plans': self.used_plans[:], 'last_error': deepcopy(self.last_error)}

    def _anonymous(self, rows, fields):
        return [dict(asset=self.aliases[r['permno']], **{k: r[k] for k in fields}) for r in rows]

    def dispatch(self, name, arguments):
        try:
            schemas = {s['name']: s for s in tool_schemas(self.policy)}
            require(name in schemas, 'UNKNOWN_TOOL', 'tool is not available')
            require(type(arguments) is dict and set(arguments) == set(schemas[name]['parameters']['properties']),
                    'ARGUMENTS', 'arguments must exactly match the tool schema')
            require(self.report is None, 'STATE', 'portfolio already finalized')
            result = getattr(self, name)(**arguments)
            self.last_error = None
            return {'ok': True, 'data': result, 'error': None}
        except (ToolError, ValueError, KeyError, TypeError) as exc:
            error = {'code': getattr(exc, 'code', 'INVALID_DATA'), 'message': str(exc)}
            self.last_error = error
            return {'ok': False, 'data': None, 'error': error}

    def get_predictions(self):
        self.reads.add('get_predictions')
        return self._anonymous(self.rows.values(), ['predicted_excess_return'])

    def get_company_context(self):
        self.reads.add('get_company_context')
        return self._anonymous(self.rows.values(), ['beta_60m', 'market_cap', 'dollar_volume'])

    def get_previous_holdings(self):
        self.reads.add('get_previous_holdings')
        return {'kind': self.previous['kind'],
                'weights': self._anonymous(self.previous_weights, ['weight'])}

    def rank_candidates(self, plan):
        require(self.reads == {'get_predictions', 'get_company_context', 'get_previous_holdings'},
                'STATE', 'read all three inputs before ranking')
        # Invalidate any earlier solution even when a replacement selection fails.
        self.candidates = self.weights = self.report = self.check_report = None
        self.checked = False
        plans = {p.name: p for p in self.policy.plans}
        require(isinstance(plan, str) and plan in plans, 'ARGUMENTS', 'unapproved candidate plan')
        selected = plans[plan]
        self.used_plans.append(plan)
        rows = [r for r in self.rows.values() if r['dollar_volume'] >= selected.min_dollar_volume
                and abs(r['beta_60m']) <= selected.max_abs_beta]
        rows.sort(key=lambda r: (-r['predicted_excess_return'], r['permno']))
        require(len(rows) >= selected.n_long + selected.n_short, 'CANDIDATES', 'insufficient eligible names')
        self.candidates = [dict(r, side='long') for r in rows[:selected.n_long]] + [
            dict(r, side='short') for r in rows[-selected.n_short:]]
        self.plan_name = plan
        return {'plan': plan, 'candidates': self._anonymous(self.candidates,
                ['predicted_excess_return', 'beta_60m', 'side'])}

    def optimize_weights(self):
        require(self.candidates is not None, 'STATE', 'rank candidates first')
        require(self.attempts < self.policy.max_optimizer_retries + 1, 'RETRY_LIMIT', 'optimizer budget exhausted')
        self.attempts += 1
        self.weights = self.check_report = None
        self.checked = False
        try:
            result = self.optimizer(candidates=deepcopy(self.candidates),
                                    previous_weights=deepcopy(self.previous_weights), policy=asdict(self.policy))
        except Exception as exc:
            # Adapter internals can contain sensitive paths/data; do not send exception bodies to an LLM.
            raise ToolError('OPTIMIZER_EXCEPTION', f'optimizer raised {type(exc).__name__}') from exc
        if not isinstance(result, dict) or result.get('status') != 'optimal':
            reason = result.get('failure_reason') if isinstance(result, dict) else None
            safe_reason = reason if reason in ('infeasible', 'unbounded', 'solver_error', 'timeout', 'missing_data') else 'unspecified'
            raise ToolError('OPTIMIZER_FAILED', f'optimizer failed: {safe_reason}; revise approved plan or stop')
        require(type(result.get('weights')) is list and bool(result['weights']), 'OPTIMIZER_INVALID', 'missing weights')
        lookup = {r['permno']: r for r in self.candidates}
        rows, seen = [], set()
        for item in result['weights']:
            identifier = self._id(item['permno'])
            require(identifier in lookup and identifier not in seen, 'OPTIMIZER_INVALID', 'unknown/duplicate weight ID')
            seen.add(identifier)
            weight = finite(item['weight'], 'weight')
            side = lookup[identifier]['side']
            require(weight >= 0 if side == 'long' else weight <= 0, 'OPTIMIZER_INVALID', 'wrong position sign')
            rows.append({'permno': identifier, 'target_month': self.month.isoformat(),
                         'weight': weight, 'beta_60m': lookup[identifier]['beta_60m']})
        # Omitted candidates are allowed, but the actual nonzero count is checked below.
        self.weights = rows
        return {'status': 'optimal_unchecked', 'weights': self._anonymous(rows, ['weight'])}

    def check_constraints(self):
        require(self.weights is not None, 'STATE', 'no optimizer solution to check')
        result = check_constraints(deepcopy(self.weights), self.policy.limits)
        result['team_checks']['net_target'] = abs(result['net']) <= self.policy.net_target_tol + self.policy.limits.comparison_tolerance
        result['configured_checks_pass'] = result['official_numeric_pass'] and all(result['team_checks'].values())
        self.check_report = result
        self.checked = result['configured_checks_pass']
        require(self.checked, 'CONSTRAINTS', json.dumps({'official': result['official_numeric_checks'],
                                                        'team': result['team_checks']}, sort_keys=True))
        return {k: v for k, v in result.items() if k not in ('target_month', 'scope')}

    def write_portfolio_report(self):
        require(self.checked, 'STATE', 'successful constraint check required')
        self.check_constraints()  # Recheck before finalization, never trust a cached success flag alone.
        r = self.check_report
        self.report = {'summary': f"Selected plan {self.plan_name}; optimizer produced {r['holdings_count']} active positions. "
                       f"Gross {r['gross']:.6f}, net {r['net']:.6f}, input beta {r['input_beta']}. "
                       'All configured rebalance checks passed. This is not realized-beta certification.',
                       'plan': self.plan_name, 'optimizer_attempts': self.attempts,
                       'top_holdings': sorted(deepcopy(self.weights), key=lambda x: (-abs(x['weight']), x['permno']))[:10],
                       'evidence': ['get_predictions', 'get_company_context', 'get_previous_holdings',
                                    'rank_candidates', 'optimize_weights', 'check_constraints']}
        return {'finalized': True, 'summary': self.report['summary']}
