"""portfolio: hand calculations and deliberately invalid holdings."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from src.portfolio.risk import ConstraintLimits, check_constraints


def rows(weights, betas=None):
    return [dict(permno=i+1, target_month='2021-01-01', weight=w,
                 beta_60m=1 if betas is None else betas[i]) for i, w in enumerate(weights)]


def test_balanced_100_percent_each_side():
    r = check_constraints(rows([.02]*50 + [-.02]*50))
    assert r['gross'] == 2 and r['net'] == 0
    assert r['holdings_count'] == 100 and r['official_numeric_pass']


def test_150_long_100_short_exceeds_gross():
    r = check_constraints(rows([.03]*50 + [-.02]*50))
    assert r['gross'] == 2.5 and r['net'] == pytest.approx(.5)
    assert not r['official_numeric_checks']['gross']


def test_candidates_are_not_holdings():
    r = check_constraints(rows([.01]*40 + [-.01]*40 + [0]*220))
    assert r['candidate_count'] == 300 and r['holdings_count'] == 80
    assert not r['official_numeric_pass']


def test_zero_weights():
    r = check_constraints(rows([0]*300))
    assert r['gross'] == r['net'] == r['holdings_count'] == 0
    assert r['official_numeric_checks']['gross']
    assert not r['official_numeric_checks']['holdings_count']


def test_equal_money_not_equal_beta():
    r = check_constraints(rows([.02]*50 + [-.02]*50, [1.5]*50 + [.5]*50),
                          ConstraintLimits(beta_tolerance=.1))
    assert r['net'] == 0 and r['input_beta'] == pytest.approx(1)
    assert r['official_numeric_pass'] and not r['team_checks']['input_beta']


def test_missing_beta_not_zero():
    r = check_constraints(rows([.02]*50+[-.02]*50, [None]*100), ConstraintLimits(beta_tolerance=.1))
    assert r['input_beta'] is None and r['missing_beta_count'] == 100
    assert not r['configured_checks_pass']


def test_tiny_weights_still_in_exposures():
    r = check_constraints(rows([1e-8, 1.00001e-8, -1e-8]))
    assert r['holdings_count'] == 1
    assert r['gross'] == pytest.approx(3.00001e-8)


@pytest.mark.parametrize('n,passed', [(99,False),(100,True),(500,True),(501,False)])
def test_count_boundaries(n, passed):
    r = check_constraints(rows([.001]*n))
    assert r['official_numeric_checks']['holdings_count'] == passed


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None, True, 'oops'])
def test_bad_weights(value):
    with pytest.raises(ValueError):
        check_constraints(rows([value]))


@pytest.mark.parametrize('weights,passed', [([.5],True),([-.5],True),([.50001],False),([-.50001],False)])
def test_net_boundaries(weights, passed):
    assert check_constraints(rows(weights))['official_numeric_checks']['net'] == passed


def test_duplicate_and_mixed_month_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        check_constraints(rows([.1])*2)
    r = rows([.1, -.1]); r[1]['target_month'] = '2021-02-01'
    with pytest.raises(ValueError, match='one month'):
        check_constraints(r)
    r[1]['target_month'] = '2021-01-31'
    with pytest.raises(ValueError, match='month start'):
        check_constraints(r)


def test_team_cap_is_opt_in():
    r = rows([.03]*10 + [.01]*40 + [-.014]*50)
    assert check_constraints(r)['official_numeric_pass']
    assert not check_constraints(r, ConstraintLimits(max_position=.02))['configured_checks_pass']


@pytest.mark.parametrize('kwargs', [dict(min_holdings=501),dict(max_holdings=-1),
    dict(position_epsilon=-1),dict(beta_tolerance=float('nan')),dict(net_min=1,net_max=0)])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        ConstraintLimits(**kwargs)


def test_cli_checks_full_period_and_exit_codes(tmp_path):
    file = tmp_path/'holdings.csv'
    with file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['permno','target_month','weight','beta_60m'])
        writer.writeheader(); writer.writerows(rows([.02]*50+[-.02]*50))
    cli = [sys.executable, str(Path(__file__).resolve().parents[1]/'check_constraints.py'), str(file)]
    one = subprocess.run(cli, capture_output=True, text=True)
    assert one.returncode == 0
    full = subprocess.run(cli+['--require-full-period'], capture_output=True, text=True)
    assert full.returncode == 1 and len(json.loads(full.stdout)['coverage']['missing_months']) == 67
    file.write_text('permno,target_month,weight\n1,2021-01-01,nan\n')
    assert subprocess.run(cli, capture_output=True).returncode == 2
