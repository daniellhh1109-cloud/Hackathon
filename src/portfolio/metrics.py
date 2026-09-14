import math
from statistics import mean, stdev
from src.portfolio.risk import finite


def benchmark_return(tb3ms):
    """Annual percent (4.8) -> monthly decimal; official cash + 4% benchmark."""
    return finite(tb3ms, 'TB3MS') / 1200 + 0.04 / 12


def monthly_total_return(weights, stock_returns, rf, *, traded_notional=0,
                         transaction_cost_bps=10, annual_borrow_bps=0):
    """Frozen, month-aligned permno maps. Cash 1-net earns/pays rf.

    Idealized fully remunerated short proceeds. Missing held returns raise.
    Caller supplies total ret, not ret_exc/ret_exc_lead1m; no renormalization.
    """
    weights = {k: finite(v, 'weight') for k, v in weights.items()}
    rf = finite(rf, 'rf')
    traded_notional = finite(traded_notional, 'traded_notional')
    tc = finite(transaction_cost_bps, 'transaction_cost_bps')
    borrow = finite(annual_borrow_bps, 'annual_borrow_bps')
    if min(traded_notional, tc, borrow) < 0:
        raise ValueError('costs and traded notional must be nonnegative')
    pnl = 0.0
    for key, weight in weights.items():
        if weight == 0:
            continue
        if key not in stock_returns:
            raise ValueError(f'missing held return: {key}')
        ret = finite(stock_returns[key], 'ret')
        if ret < -1:
            raise ValueError('stock return below -100%')
        pnl += weight * ret
    cash = (1 - math.fsum(weights.values())) * rf
    transaction_cost = traded_notional * tc / 10000
    borrow_cost = math.fsum(-w for w in weights.values() if w < 0) * borrow / 10000 / 12
    return {'stock_pnl': pnl, 'cash_return': cash, 'transaction_cost': transaction_cost,
            'borrow_cost': borrow_cost, 'total_return': pnl + cash - transaction_cost - borrow_cost}


def traded_notional(target_weights, pretrade_weights):
    """Buy+sell dollars / pretrade NAV; prior weights must include drift.

    Both maps use the same NAV denominator. Initial pretrade map is {}.
    Half this value is half-L1 turnover, not the cost base used here.
    """
    target = {k: finite(v, 'weight') for k, v in target_weights.items()}
    prior = {k: finite(v, 'weight') for k, v in pretrade_weights.items()}
    return math.fsum(abs(target.get(k, 0) - prior.get(k, 0)) for k in target.keys() | prior.keys())


def performance(total_returns, risk_free, tb3ms):
    """Already month-aligned ordered vectors, >=2 months, no dropna."""
    returns = [finite(v, 'total return') for v in total_returns]
    rf = [finite(v, 'rf') for v in risk_free]
    benchmark = [benchmark_return(v) for v in tb3ms]
    if len(returns) < 2 or len(returns) != len(rf) or len(returns) != len(benchmark):
        raise ValueError('equal aligned vectors with at least two months required')
    if min(returns) <= -1:
        raise ValueError('NAV must stay strictly positive')
    excess = [r-f for r, f in zip(returns, rf)]
    active = [r-b for r, b in zip(returns, benchmark)]
    def ratio(values):
        sigma = stdev(values)
        return None if sigma <= 1e-15 else mean(values) / sigma * math.sqrt(12)
    nav, peak, drawdown = 1.0, 1.0, 0.0
    path = [nav]
    for ret in returns:
        nav *= 1 + ret
        peak = max(peak, nav)
        drawdown = max(drawdown, 1 - nav / peak)
        path.append(nav)
    return {'n_months': len(returns), 'cumulative_return': nav - 1,
            'cagr': nav ** (12 / len(returns)) - 1,
            'annualized_volatility': stdev(returns) * math.sqrt(12),
            'sharpe': ratio(excess), 'information_ratio': ratio(active),
            'max_drawdown': drawdown, 'worst_month_return': min(returns), 'nav': path}
