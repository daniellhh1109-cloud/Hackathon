import math
import pytest
from src.portfolio.metrics import benchmark_return, monthly_total_return, traded_notional, performance


def test_cash_cost_and_borrow_hand_calculation():
    # +100% earns 10%; -100% loses 2%; cash earns .4%; trades cost .2%; borrow .1%.
    r = monthly_total_return({'L':1,'S':-1}, {'L':.1,'S':.02}, .004,
                             traded_notional=2, annual_borrow_bps=120)
    assert r['total_return'] == pytest.approx(.081)
    assert r['transaction_cost'] == pytest.approx(.002)


def test_net_cash_denominator():
    assert monthly_total_return({'L':.5}, {'L':.1}, .004)['total_return'] == pytest.approx(.052)


def test_missing_returns_not_filled_or_dropped():
    with pytest.raises(ValueError, match='missing held return'):
        monthly_total_return({'A':1}, {}, 0)
    with pytest.raises(ValueError):
        monthly_total_return({'A':1}, {'A':float('nan')}, 0)
    assert monthly_total_return({'A':0}, {}, .004)['total_return'] == .004


def test_turnover_full_replacement_sign_flip_and_initial():
    assert traded_notional({'B':1}, {'A':1}) == 2
    assert traded_notional({'A':-1}, {'A':1}) == 2
    assert traded_notional({'A':1,'B':-1}, {}) == 2
    assert traded_notional({'A':.5}, {'A':.6}) == pytest.approx(.1)


def test_benchmark_sharpe_ir_hand_calculation():
    assert benchmark_return(4.8) == pytest.approx(.004 + .04/12)
    r = performance([.01,.02], [.004,.004], [4.8,4.8])
    assert r['sharpe'] == pytest.approx(.011/(.01/math.sqrt(2))*math.sqrt(12))
    assert r['information_ratio'] == pytest.approx((.015-.004-.04/12)/(.01/math.sqrt(2))*math.sqrt(12))


def test_first_month_loss_drawdown():
    r = performance([-.1,0], [0,0], [0,0])
    assert r['nav'] == [1,.9,.9] and r['max_drawdown'] == pytest.approx(.1)


def test_peak_to_trough():
    assert performance([.1,-.2,.1],[0]*3,[0]*3)['max_drawdown'] == pytest.approx(.2)


def test_constant_returns_undefined_ratios():
    r = performance([.01,.01], [0,0], [0,0])
    assert r['sharpe'] is None and r['information_ratio'] is None


@pytest.mark.parametrize('returns', [[-1,0],[-1.1,0],[float('nan'),0]])
def test_invalid_nav(returns):
    with pytest.raises(ValueError):
        performance(returns,[0,0],[0,0])


def test_misaligned_lengths():
    with pytest.raises(ValueError):
        performance([0,0],[0],[0,0])
