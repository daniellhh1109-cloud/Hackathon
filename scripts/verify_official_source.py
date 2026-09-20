"""Compare actual original source blocks with the adaptation on complete-label fixtures.

Only executes the reviewed preprocessing and estimator AST nodes, not original
path I/O or old-date loops. Separate raw-attempt logs cover unchanged execution.
"""
import argparse
import ast
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error
from src.data.build_samples import build_baseline_panel
from src.models.official_linear_baseline import dense_monthly_transform,fit_official_year
from scripts.run_member_d import sha256,write_json


class AppendCompatibility(ast.NodeTransformer):
    """Only replace removed DataFrame._append with public pd.concat in memory."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == '_append':
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id='pd', ctx=ast.Load()), attr='concat', ctx=ast.Load()),
                args=[ast.List(elts=[node.func.value, node.args[0]], ctx=ast.Load())],
                keywords=node.keywords), node)
        return node


def execute_nodes(nodes, environment):
    module=AppendCompatibility().visit(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(ast.fix_missing_locations(module),'<reviewed official source block>','exec'),environment)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='/Users/daniel/Downloads/penalized_linear_hackathon.py')
    parser.add_argument('--output',default='outputs/member_d_official/source_equivalence.json')
    args=parser.parse_args()
    tree=ast.parse(Path(args.source).read_text())
    body=next(n.body for n in tree.body if isinstance(n,ast.If))
    # Source anchors deliberately fail if a new source version changes the structure.
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(body) if assignment(n,'monthly'))
    end=next(i for i,n in enumerate(body) if assignment(n,'starting'))
    yearly=next(n for n in body if isinstance(n,ast.While)).body
    fit_start=next(i for i,n in enumerate(yearly) if assignment(n,'scaler'))
    fit_end=next(i for i,n in enumerate(yearly) if assignment(n,'pred_out'))
    factors=[f'f{i}' for i in range(147)]
    rows=[]
    for month in ['2018-11-30','2018-12-31','2020-12-31']:
        for i in range(10):
            rows.append({'permno':i+1,'date':month,'eom':month,'ret_exc_lead1m':.01*(i-4),
                         **{f:float((i+j)%5) for j,f in enumerate(factors)}})
    panel=build_baseline_panel(pd.DataFrame(rows))
    x,_=dense_monthly_transform(panel,factors)
    adapted,report,_=fit_official_year(panel,x,factors,2021)
    # Legacy 'date' means holding month. All fixture labels are finite, so original
    # label-based dropping does not change membership in this equivalence check.
    legacy=panel.copy(); legacy['date']=pd.to_datetime(legacy.target_month)
    legacy['year']=legacy.date.dt.year; legacy['month']=legacy.date.dt.month
    legacy['stock_exret']=legacy.ret_exc_lead1m
    env={'np':np,'pd':pd,'new_set':legacy,'stock_vars':factors,
         'StandardScaler':StandardScaler,'LinearRegression':LinearRegression,'Lasso':Lasso,
         'Ridge':Ridge,'ElasticNet':ElasticNet,'mean_squared_error':mean_squared_error,'ret_var':'stock_exret'}
    execute_nodes(body[start:end],env)
    official=env['data']
    np.testing.assert_allclose(official[factors],x,atol=1e-14)
    env.update(train=official.iloc[:10].copy(),validate=official.iloc[10:20].copy(),test=official.iloc[20:].copy())
    execute_nodes(yearly[fit_start:fit_end],env)
    errors={m:float(np.max(np.abs(env['reg_pred'][m].to_numpy()-adapted[m].to_numpy()))) for m in ['ols','lasso','ridge','en']}
    if any(v>1e-10 for v in errors.values()): raise AssertionError(errors)
    result={'official_source':args.source,'sha256':sha256(args.source),'fixture_rows':len(panel),'factors':147,
            'dense_ranks_match':True,'compatibility_change':'DataFrame._append replaced in memory by pd.concat; source file unchanged','prediction_max_abs_difference':errors,
            'rank_source_lines':[body[start].lineno,body[end-1].end_lineno],
            'fit_source_lines':[yearly[fit_start].lineno,yearly[fit_end-1].end_lineno],
            'scope':'actual original AST blocks, all original alpha grids; complete labels, C-aligned fixture periods; not unchanged full-script execution'}
    write_json(Path(args.output),result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
