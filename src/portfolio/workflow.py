"""Pipeline integration: monthly decisions commit before realized outcome access."""
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from src.agent.controllers import MockController,OpenAIController
from src.agent.pipeline import months_between,policy_from_dict
from src.agent.portfolio_agent import run_month,write_json
from src.agent.tools import PortfolioTools,digest
from src.portfolio.context import ContextStore,read_prediction,audit_prediction_coverage
from src.portfolio.optimizer import optimize_weights
from src.portfolio.backtest import OutcomeStore,read_benchmarks,summarize


def run_backtest(config,output_dir):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    completed=[];returns=[];holding_rows=[];audits=[];previous={};previous_hash=None
    try:
        policy=policy_from_dict(config.get('policy',{}))
        months=months_between(config['start_month'],config['end_month'])
        coverage=audit_prediction_coverage(config['predictions_dir'],months)
        write_json(output/'prediction_coverage.json',coverage)
        context=ContextStore(config['raw_path'])
        outcomes=OutcomeStore(config['raw_path'])
        # Benchmarks are evaluated outside controller context, never candidate inputs.
        benchmarks=read_benchmarks(config['tb3ms_path'],config['sp500_path'])
        for m in months:
            key=pd.Timestamp(m)
            if key not in benchmarks.index or benchmarks.loc[key].isna().any():
                raise ValueError(f'missing benchmark month {m}; include prior December market close')
        write_json(output/'config.json',config)
        for date in months:
            month=date[:7];period=pd.Period(month,freq='M')
            predictions,path=read_prediction(config['predictions_dir'],month)
            p,c,labels,audit=context.prepare(predictions,month,**config.get('filters',{}))
            audit['month']=month;audits.append(audit)
            state=dict(target_month=date,as_of=(period-1).end_time.date().isoformat(),
                       kind='initial' if not completed else 'drifted',
                       weights=[{'permno':k,'weight':v} for k,v in previous.items()])
            if previous_hash:state['source_holdings_sha256']=previous_hash
            cc=config.get('controller',{'kind':'mock'})
            if cc['kind']=='mock':controller=MockController()
            elif cc['kind']=='openai':controller=OpenAIController(**{k:v for k,v in cc.items() if k!='kind'})
            else:raise ValueError('unknown controller kind')
            tools=PortfolioTools(date,p,c,state,optimize_weights,policy)
            directory=output/month
            result=run_month(tools,controller,directory)
            write_json(directory/'eligibility.json',audit)
            if result['status']!='committed':raise ValueError(f'agent failed in {month}: {result}')
            committed=json.loads((directory/'holdings.json').read_text())
            previous_hash=result['holdings_sha256']
            # Only here may this month's actual stock returns enter the evaluation stage.
            bench=benchmarks.loc[pd.Timestamp(date)]
            row,previous=outcomes.evaluate(directory,previous,float(bench.rf),**config.get('costs',{}))
            row.update({k:float(bench[k]) for k in ('rf','benchmark','market','TB3MS')})
            checks=json.loads((directory/'constraint_report.json').read_text())
            row.update(gross=checks['gross'],net=checks['net'],holdings_count=checks['holdings_count'],
                       input_beta=checks['input_beta'],max_abs_weight=checks['max_abs_weight'])
            row['top10_concentration']=sum(sorted([abs(r['weight']) for r in committed],reverse=True)[:10])
            lookup={r['permno']:r for r in c};short=[r for r in committed if r['weight']<0]
            row['short_min_market_cap']=min(lookup[r['permno']]['market_cap'] for r in short)
            row['short_min_dollar_volume']=min(lookup[r['permno']]['dollar_volume'] for r in short)
            # R2 uses labels only after holdings are locked. Missing labels never determine universe.
            target=pd.read_parquet(config['raw_path'],columns=['permno','eom','ret_exc_lead1m'],
                   filters=[('eom','==',(period-1).end_time.normalize())])
            scored=predictions.merge(target[['permno','ret_exc_lead1m']],on='permno',validate='one_to_one')
            scored=scored.dropna(subset=['ret_exc_lead1m'])
            row['prediction_sse']=float(((scored.predicted_excess_return-scored.ret_exc_lead1m)**2).sum())
            row['prediction_sst_zero']=float((scored.ret_exc_lead1m**2).sum())
            row['prediction_scored']=len(scored)
            returns.append(row)
            for r in committed:
                if abs(r['weight'])<=policy.limits.position_epsilon:continue
                names=labels[r['permno']]
                holding_rows.append({'Date':date,'PERMNO':r['permno'],'TICKER':names['TICKER'],
                    'COMPANY NAME':names['COMPANY NAME'],'WEIGHT':100*r['weight']})
            write_json(directory/'evaluation.json',row)
            write_json(directory/'drifted_weights.json',previous)
            completed.append(month)
            pd.DataFrame(returns).to_csv(output/'portfolio_returns.csv',index=False)
            pd.DataFrame(holding_rows).to_csv(output/'monthly_holdings.csv',index=False)
        table=pd.DataFrame(returns)
        metrics,timeline=summarize(table)
        denom=table.prediction_sst_zero.sum()
        metrics['oos_r2_zero']=None if denom==0 else float(1-table.prediction_sse.sum()/denom)
        metrics.update(average_gross=float(table.gross.mean()),max_gross=float(table.gross.max()),
                       min_net=float(table.net.min()),max_net=float(table.net.max()),
                       mean_traded_notional=float(table.traded_notional.mean()),
                       borrow_availability_verified=False)
        write_json(output/'metrics.json',metrics)
        pd.DataFrame([metrics]).to_csv(output/'metrics.csv',index=False)
        timeline.to_csv(output/'timeline.csv',index=False)
        pd.DataFrame(audits).to_json(output/'eligibility.json',orient='records',indent=2)
        years=pd.to_datetime(table.Date).dt.year
        table.groupby(years).total_return.apply(lambda x:float((1+x).prod()-1)).rename('total_return').to_csv(output/'annual_returns.csv')
        incomplete_names=any(not r['TICKER'] or not r['COMPANY NAME'] for r in holding_rows)
        result={'status':'completed','synthetic':config.get('synthetic',False),'months':completed,'full_oos':len(months)==68,
                'names_complete':not incomplete_names,'controller_kind':config.get('controller',{}).get('kind','mock'),
                'holdings_weight_unit':'percent_of_NAV',
                'submission_ready':False,'remaining_manual_checks':['CVs','deck review','official submission requirements','borrow feasibility'],
                'costs':config.get('costs',{}),'cash_assumption':'1-net earns/pays TB3MS/1200; fully remunerated short proceeds',
                'missing_return_policy':'error; no label-based reselection or zero fill'}
        write_json(output/'run_result.json',result)
        return result
    except Exception as exc:
        write_json(output/'run_result.json',{'status':'failed','months_completed':completed,'error':str(exc)})
        raise
