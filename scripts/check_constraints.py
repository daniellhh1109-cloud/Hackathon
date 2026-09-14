"""CSV CLI: python check_constraints.py holdings.csv --output report.json."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from src.portfolio.risk import ConstraintLimits, check_constraints
from src.utils.dates import add_months


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('holdings', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--beta-tolerance', type=float)
    parser.add_argument('--max-position', type=float)
    parser.add_argument('--require-full-period', action='store_true')
    args = parser.parse_args()
    try:
        limits = ConstraintLimits(beta_tolerance=args.beta_tolerance, max_position=args.max_position)
        grouped = defaultdict(list)
        with args.holdings.open(newline='') as stream:
            for row in csv.DictReader(stream):
                grouped[row['target_month']].append(row)
        if not grouped:
            raise ValueError('empty holdings CSV')
        reports = [check_constraints(grouped[k], limits) for k in sorted(grouped)]
        expected = {add_months('2021-01-01', i).isoformat() for i in range(68)}
        actual = {r['target_month'] for r in reports}
        passed = all(r['configured_checks_pass'] for r in reports)
        if args.require_full_period:
            passed = passed and actual == expected
        result = {'passed': passed, 'full_period_required': args.require_full_period,
                  'coverage': {'missing_months': sorted(expected-actual), 'extra_months': sorted(actual-expected)},
                  'months': reports}
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Invalid input: {exc}\n')
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
