"""CLI orchestration for A's decision stage and optional post-commit D adapter."""
import argparse
import importlib
import json
from pathlib import Path

import yaml

from src.agent.controllers import MockController, OpenAIController
from src.agent.portfolio_agent import run_month, write_json
from src.agent.tools import AgentPolicy, CandidatePlan, PortfolioTools, digest
from src.portfolio.risk import ConstraintLimits
from src.utils.dates import add_months, as_date, month_start


def load_callable(spec):
    if not isinstance(spec, str) or spec.count(':') != 1:
        raise ValueError('adapter must be module:function')
    module, name = spec.split(':')
    if not all(p.isidentifier() for p in module.split('.')) or not name.isidentifier():
        raise ValueError('invalid adapter identifier')
    value = getattr(importlib.import_module(module), name, None)
    if not callable(value):
        raise ValueError(f'Adapter unavailable: {spec}. Complete the responsible member module first.')
    if getattr(value, '__module__', '') == 'src.agent.demo':
        raise ValueError('synthetic adapters are forbidden in production mode')
    return value


def load_table(path):
    path = Path(path)
    if path.suffix == '.json':
        result = json.loads(path.read_text())
    elif path.suffix in ('.csv', '.parquet'):
        import pandas as pd
        frame = pd.read_csv(path, dtype={'permno': str}) if path.suffix == '.csv' else pd.read_parquet(path)
        # Timestamp metadata is normalized; no return columns are made visible to tools.
        for key in ('target_month', 'information_cutoff', 'model_selection_end', 'available_at'):
            if key in frame:
                frame[key] = pd.to_datetime(frame[key], errors='raise').dt.strftime('%Y-%m-%d')
        result = frame.to_dict('records')
    else:
        raise ValueError('tables must be JSON, CSV or Parquet')
    if not isinstance(result, list):
        raise ValueError('table must be a list of records')
    return result


def months_between(start, end):
    start, end = as_date(start), as_date(end)
    if start != month_start(start) or end != month_start(end) or end < start:
        raise ValueError('start/end must be ordered ISO month starts')
    if start < as_date('2021-01-01') or end > as_date('2026-08-01'):
        raise ValueError('requested months outside competition OOS interval')
    months = []
    while start <= end:
        months.append(start.isoformat())
        start = add_months(start, 1)
    return months


def policy_from_dict(raw):
    raw = dict(raw)
    if 'plans' in raw:
        raw['plans'] = tuple(CandidatePlan(**item) for item in raw['plans'])
    if 'limits' in raw:
        raw['limits'] = ConstraintLimits(**raw['limits'])
    return AgentPolicy(**raw)


def evaluate_committed(directory, evaluator):
    """Called only after all requested decisions. Evaluation never returns to a controller."""
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    path = directory / 'holdings.json'
    if result.get('status') != 'committed' or digest(json.loads(path.read_text())) != result['holdings_sha256']:
        raise ValueError('cannot evaluate an uncommitted or modified portfolio')
    evaluation_dir = directory / 'evaluation'
    evaluation_dir.mkdir(exist_ok=False)
    evaluation = evaluator(frozen_holdings_path=path.resolve(), output_dir=evaluation_dir.resolve())
    if digest(json.loads(path.read_text())) != result['holdings_sha256']:
        raise ValueError('evaluator modified frozen holdings')
    write_json(evaluation_dir / 'adapter_result.json', evaluation)
    return evaluation


def execute(config, output_dir, *, smoke=False):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    stage = 'preflight'
    committed = []
    try:
        policy = policy_from_dict(config.get('policy', {}))
        months = months_between(config['start_month'], config['end_month'])
        controller_config = config.get('controller', {'kind': 'mock'})
        controller_kind = controller_config.get('kind')
        if controller_kind not in ('mock', 'openai'):
            raise ValueError('controller kind must be mock or openai')
        if smoke:
            from src.agent.demo import components, equal_weight_fixture, evaluate_fixture
            optimizer, evaluator = equal_weight_fixture, evaluate_fixture
        else:
            optimizer = load_callable(config['optimizer'])
            evaluator = load_callable(config['evaluator']) if config.get('evaluator') else None
        # Resolve API configuration before any monthly artifacts are produced.
        def make_controller():
            if controller_kind == 'mock':
                return MockController()
            return OpenAIController(**{k: v for k, v in controller_config.items() if k != 'kind'})
        controller = make_controller()
        write_json(directory / 'run_manifest.json', {'synthetic': smoke, 'config': config,
            'controller': controller.metadata, 'months': months,
            'scope': 'Portfolio decision orchestration. Optional evaluation is a separate trusted adapter.'})
        previous_hash = None
        for index, month in enumerate(months):
            stage = f'decision:{month}'
            if smoke:
                predictions, context, previous = components(month, initial=index == 0)
                if committed:
                    prior = json.loads((committed[-1] / 'holdings.json').read_text())
                    previous['weights'] = [{'permno': r['permno'], 'weight': r['weight']} for r in prior]
                    previous['source_holdings_sha256'] = previous_hash
                    # Synthetic zero prior returns => target and drifted weights coincide.
            else:
                item = config['snapshots'][month]
                predictions, context = load_table(item['predictions']), load_table(item['context'])
                previous = json.loads(Path(item['previous']).read_text())
            if index and (previous['kind'] != 'drifted' or previous.get('source_holdings_sha256') != previous_hash):
                raise ValueError('previous state must reference the preceding committed holdings hash')
            if not index and previous['kind'] == 'initial' and config.get('initial_portfolio') is not True:
                raise ValueError('empty initial portfolio must be explicitly declared')
            if not index and previous['kind'] == 'drifted':
                source = Path(config['previous_committed_dir'])
                old_result = json.loads((source / 'result.json').read_text())
                old_rows = json.loads((source / 'holdings.json').read_text())
                if (old_result.get('status') != 'committed'
                        or old_result.get('target_month') != add_months(month, -1).isoformat()
                        or old_result.get('holdings_sha256') != digest(old_rows)
                        or previous.get('source_holdings_sha256') != digest(old_rows)):
                    raise ValueError('continuation prior holdings provenance mismatch')
            tools = PortfolioTools(month, predictions, context, previous, optimizer, policy)
            month_dir = directory / month[:7]
            result = run_month(tools, make_controller(), month_dir)
            if result['status'] != 'committed':
                raise RuntimeError(f"monthly agent failed ({result['error']['code']}); see {month[:7]}/result.json")
            previous_hash = result['holdings_sha256']
            committed.append(month_dir)
        # All decisions are locked before D can read outcome data. No retry on metrics.
        stage = 'evaluation'
        if evaluator is not None:
            for month_dir in committed:
                evaluate_committed(month_dir, evaluator)
        summary = {'status': 'completed', 'synthetic': smoke, 'months_committed': len(committed),
                   'evaluation': 'adapter_completed' if evaluator else 'not_requested',
                   'controller': controller_kind,
                   'competition_submission_ready': False}
        write_json(directory / 'run_result.json', summary)
        return summary
    except Exception as exc:
        summary = {'status': 'failed', 'stage': stage, 'synthetic': smoke,
                   'months_committed': len(committed), 'error_type': type(exc).__name__,
                   'message': str(exc), 'competition_submission_ready': False}
        write_json(directory / 'run_result.json', summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/portfolio_agent.yaml')
    parser.add_argument('--output-dir', required=True, help='New directory; never overwrites previous runs')
    parser.add_argument('--smoke', action='store_true', help='Synthetic data and optimizer/evaluator fixtures only')
    args = parser.parse_args(argv)
    if args.smoke:
        config = {'start_month': '2021-01-01', 'end_month': '2021-02-01',
                  'initial_portfolio': True, 'controller': {'kind': 'mock'}}
    else:
        config = yaml.safe_load(Path(args.config).read_text())
    result = execute(config, args.output_dir, smoke=args.smoke)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed' else 2
