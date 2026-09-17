"""Member C: convex long/short optimizer with explicit active-position floor."""
import math
import numpy as np
from src.portfolio.risk import ConstraintLimits, check_constraints, finite


def optimize_weights(*, candidates, previous_weights, policy):
    """Fixed signed candidates; CVXPY/CLARABEL solves continuous magnitudes.

    Positive minimum magnitude makes every selected name an actual holding.
    Exit trades contribute a constant to L1 turnover. Never relax constraints.
    """
    import cvxpy as cp
    try:
        limits = ConstraintLimits(**policy['limits'])
        n = len(candidates)
        if not limits.min_holdings <= n <= limits.max_holdings:
            return {'status': 'failed', 'failure_reason': 'infeasible'}
        ids = [str(r['permno']) for r in candidates]
        if len(set(ids)) != n or any(r['side'] not in ('long','short') for r in candidates):
            raise ValueError('duplicate candidates or invalid side')
        if len({r['target_month'] for r in candidates}) != 1:
            raise ValueError('mixed candidate months')
        sides = np.array([1. if r['side']=='long' else -1. for r in candidates])
        if len(set(sides)) != 2: raise ValueError('both books required')
        alpha = np.array([finite(r['predicted_excess_return'],'prediction') for r in candidates])
        beta = np.array([finite(r['beta_60m'],'beta') for r in candidates])
        old={}
        for r in previous_weights:
            identifier=str(r['permno'])
            if identifier in old: raise ValueError('duplicate previous position')
            old[identifier]=finite(r['weight'],'previous weight')
        floor=finite(policy.get('min_position',.001),'min_position')
        gross=finite(policy.get('gross_target',2.),'gross_target')
        cap=limits.max_position if limits.max_position is not None else limits.gross_limit
        if not limits.position_epsilon < floor <= cap or not 0 < gross <= limits.gross_limit:
            raise ValueError('invalid floor/gross target')
        if n*floor>gross or n*cap<gross:
            return {'status':'failed','failure_reason':'infeasible'}
        turnover_penalty=finite(policy['lambda_turnover'],'lambda_turnover')
        l2=finite(policy['lambda_l2'],'lambda_l2')
        net_tol=finite(policy['net_target_tol'],'net_target_tol')
        if min(turnover_penalty,l2,net_tol)<0: raise ValueError('negative penalty/tolerance')
        x=cp.Variable(n)
        w=cp.multiply(sides,x)
        prior=np.array([old.get(i,0.) for i in ids])
        exits=math.fsum(abs(v) for k,v in old.items() if k not in set(ids))
        turnover=cp.norm1(w-prior)+exits
        constraints=[x>=floor,x<=cap,cp.sum(x)==gross,
                     cp.sum(w)>=limits.net_min,cp.sum(w)<=limits.net_max,
                     cp.abs(cp.sum(w))<=net_tol]
        if limits.beta_tolerance is not None:
            constraints.append(cp.abs(beta@w)<=limits.beta_tolerance)
        problem=cp.Problem(cp.Maximize(alpha@w-turnover_penalty*turnover-l2*cp.sum_squares(w)),constraints)
        problem.solve(solver='CLARABEL',max_iter=300,time_limit=30.,
                      tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        if problem.status!='optimal' or w.value is None:
            return {'status':'failed','failure_reason': 'infeasible' if 'infeasible' in str(problem.status) else 'solver_error'}
        values=np.asarray(w.value).reshape(-1)
        rows=[dict(permno=i,target_month=candidates[j]['target_month'],weight=float(values[j]),beta_60m=float(beta[j])) for j,i in enumerate(ids)]
        report=check_constraints(rows,limits)
        if not report['configured_checks_pass'] or abs(report['net'])>net_tol+limits.comparison_tolerance:
            return {'status':'failed','failure_reason':'solver_error'}
        if np.min(np.abs(values)) < floor-1e-8:
            return {'status':'failed','failure_reason':'solver_error'}
        return {'status':'optimal','weights':[{'permno':r['permno'],'weight':r['weight']} for r in rows],
                'solver':'CLARABEL','objective_value':float(problem.value),
                'traded_notional':float(np.abs(values-prior).sum()+exits),
                'exit_traded_notional':exits,'constraint_report':report}
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'failed','failure_reason':'missing_data','detail':str(exc)}
    except cp.error.SolverError:
        return {'status':'failed','failure_reason':'solver_error'}
