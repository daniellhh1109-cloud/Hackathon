"""Bounded orchestration, auditable tool logs, and immutable per-month output."""
from copy import deepcopy
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path

from src.agent.prompts import SYSTEM_PROMPT, PROMPT_VERSION
from src.agent.tools import digest, tool_schemas


def write_json(path, data):
    path = Path(path)
    # New run directories + exclusive writes prevent accidental overwrite/resume.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def run_month(tools, controller, output_dir):
    """Controller only sees snapshots and anonymous tool results, never tools itself.

    Success means a validated holdings artifact was committed, NOT that returns
    were evaluated. Consumers must require result.json status='committed'.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    schemas = tool_schemas(tools.policy)
    write_json(directory / 'manifest.json', {'target_month': tools.month.isoformat(),
        'cutoff': tools.cutoff.isoformat(), 'policy': asdict(tools.policy),
        'provenance': tools.provenance, 'controller': controller.metadata,
        'prompt_version': PROMPT_VERSION, 'system_prompt': SYSTEM_PROMPT,
        'tool_schemas': schemas})
    write_json(directory / 'identity_map.json', {alias: identifier for identifier, alias in tools.aliases.items()})
    history = []
    failure = None
    with (directory / 'tool_calls.jsonl').open('x', encoding='utf-8') as log:
        for step in range(tools.policy.max_tool_calls):
            if tools.attempts >= tools.policy.max_optimizer_retries + 1 and not tools.weights:
                failure = {'code': 'RETRY_LIMIT', 'message': 'optimizer attempt budget exhausted'}
                break
            try:
                action = controller.choose(deepcopy(tools.snapshot()), deepcopy(history), deepcopy(schemas))
                if type(action) is not dict or set(action) != {'name', 'arguments'} or not isinstance(action['name'], str):
                    raise ValueError('controller action must contain exactly name and arguments')
                # Validate serializability and forbid NaN before execution/logging.
                json.dumps(action, allow_nan=False)
                result = tools.dispatch(action['name'], action['arguments'])
            except Exception as exc:
                failure = {'code': 'CONTROLLER_ERROR', 'message': f'controller failed: {type(exc).__name__}'}
                break
            entry = {'step': step + 1, 'action': action, 'result': result}
            history.append(entry)
            log.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            if tools.report is not None:
                break
        else:
            failure = {'code': 'TOOL_LIMIT', 'message': 'tool-call budget exhausted'}
    if tools.report is None:
        failure = failure or {'code': 'NOT_FINALIZED', 'message': 'no checked portfolio'}
        result = {'status': 'failed', 'target_month': tools.month.isoformat(), 'error': failure,
                  'last_tool_error': tools.last_error, 'optimizer_attempts': tools.attempts,
                  'tool_calls': len(history)}
        if hasattr(controller, 'calls'):
            write_json(directory / 'api_calls.json', controller.calls)
        write_json(directory / 'result.json', result)
        return result
    # Commit AFTER successful report and independent check, never accept controller weights.
    rows = deepcopy(tools.weights)
    write_json(directory / 'holdings.json', rows)
    with (directory / 'holdings.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['target_month', 'permno', 'weight', 'beta_60m'])
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / 'constraint_report.json', tools.check_report)
    write_json(directory / 'portfolio_report.json', tools.report)
    result = {'status': 'committed', 'target_month': tools.month.isoformat(),
              'holdings_sha256': digest(rows), 'optimizer_attempts': tools.attempts,
              'tool_calls': len(history), 'controller': controller.metadata}
    if hasattr(controller, 'calls'):
        write_json(directory / 'api_calls.json', controller.calls)
    write_json(directory / 'result.json', result)  # Commit marker written last.
    return result
