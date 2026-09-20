"""Reproduce five required synthetic cases and metric hand calculations."""
import csv
import hashlib
import json
from pathlib import Path
from src.portfolio.risk import ConstraintLimits, check_constraints
from src.portfolio.metrics import monthly_total_return, performance


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'outputs/portfolio_verification'
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ('balanced', [.02]*50+[-.02]*50, [1]*100),
        ('gross_250_percent', [.03]*50+[-.02]*50, [1]*100),
        ('300_candidates_80_positions', [.01]*40+[-.01]*40+[0]*220, [1]*300),
        ('all_zero', [0]*300, [1]*300),
        ('net_zero_beta_one', [.02]*50+[-.02]*50, [1.5]*50+[.5]*50),
    ]
    results = {}
    for name, weights, betas in cases:
        rows = [dict(permno=i+1,target_month='2021-01-01',weight=w,beta_60m=b)
                for i,(w,b) in enumerate(zip(weights,betas))]
        with (out/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        results[name] = check_constraints(rows, ConstraintLimits(beta_tolerance=.1))
    result = {'data_kind':'synthetic hand calculations; not strategy performance',
              'cases':results,
              'cost_example':monthly_total_return({'L':1,'S':-1},{'L':.1,'S':.02},.004,
                                  traded_notional=2,annual_borrow_bps=120),
              'two_month_metrics':performance([.01,.02],[.004,.004],[4.8,4.8]),
              'first_month_loss':performance([-.1,0],[0,0],[0,0]),
              'source_sha256':{}}
    for name in ['src/portfolio/risk.py','src/portfolio/metrics.py','check_constraints.py',
                 'scripts/verify_portfolio.py','tests/test_constraints.py','tests/test_portfolio_metrics.py']:
        result['source_sha256'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(cases)} hand cases and metric examples to {out}')


if __name__ == '__main__':
    main()
