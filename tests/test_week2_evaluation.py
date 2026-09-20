"""D: metric arithmetic, common scoring rows and train/validation-only ridge."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from src.training.evaluation import regression_metrics,evaluate_table,label_scale
from src.models.window_ridge import fit_ridge,predict_ridge,feature_batches


def test_huber_and_zero_r2_hand_calculation():
    result=regression_metrics([1.,-1.,np.nan],[.5,-.5,100.],huber_delta=.25)
    assert result['oos_r2_zero']==.75
    assert result['mae']==.5 and result['rmse']==.5
    assert result['huber_loss']==pytest.approx(.09375)
    assert result['huber_linear_fraction']==1
    assert result['n_predictions']==3 and result['n_scored']==2
    assert regression_metrics([0.],[0.])['oos_r2_zero'] is None
    assert regression_metrics([np.nan],[0.])['huber_loss'] is None
    with pytest.raises(ValueError): regression_metrics([1.],[float('nan')])
    with pytest.raises(ValueError): regression_metrics([1.],[1.],huber_delta=0)


def test_same_label_rows_monthly_ic_and_constant_baseline():
    table=pd.DataFrame({'permno':[1,2,3,4,1,2,3],
        'target_month':['2021-01']*4+['2021-02']*3,
        'realized_target':[1.,2.,3.,np.nan,1.,2.,3.],
        'model':[1.,2.,3.,100.,3.,2.,1.],'zero':[0.]*7})
    summary,monthly=evaluate_table(table,['model','zero'])
    assert summary[0]['n_scored']==summary[1]['n_scored']==6
    assert summary[0]['mean_monthly_rank_ic']==pytest.approx(0.)
    assert summary[0]['rank_ic_months']==2
    assert summary[1]['mean_monthly_rank_ic'] is None
    assert summary[1]['rank_ic_months']==0
    assert [r['rank_ic'] for r in monthly[:2]]==pytest.approx([1.,-1.])
    with pytest.raises(ValueError): evaluate_table(pd.concat([table,table.iloc[:1]]),['model'])


class ToyDataset:
    def __init__(self,seed,n):
        rng=np.random.default_rng(seed)
        self.quant=rng.normal(size=(n+11,147)).astype('float32')
        self.rows=np.arange(11,n+11)
        self.labels=.04*self.quant[:,0]-.02*self.quant[:,1]+.01
    def __len__(self): return len(self.rows)


@pytest.mark.parametrize('mode',['latest','window'])
def test_streamed_ridge_matches_sklearn_and_validation_does_not_fit_coefficients(mode):
    train,val=ToyDataset(1,35),ToyDataset(2,20)
    with threadpool_limits(limits=2):
        actual,report=fit_ridge(train,val,mode=mode,alphas=[10.])
        xt=np.concatenate([x for _,x in feature_batches(train,mode)])
        xv=np.concatenate([x for _,x in feature_batches(val,mode)])
        reference=Ridge(alpha=10.,solver='cholesky').fit(xt,train.labels[train.rows])
        np.testing.assert_allclose(predict_ridge(val,actual),reference.predict(xv),atol=1e-10)
        np.testing.assert_allclose(actual['coef'],reference.coef_,atol=1e-10)
        val.labels=np.ones_like(val.labels)*20
        after,_=fit_ridge(train,val,mode=mode,alphas=[10.])
        np.testing.assert_array_equal(after['coef'],actual['coef'])
        assert after['intercept']==actual['intercept']
        assert report['selection_metric']=='validation_huber_loss'


def test_ridge_selection_uses_validation_loss():
    with threadpool_limits(limits=2):
        model,report=fit_ridge(ToyDataset(1,35),ToyDataset(2,20),mode='latest',alphas=[1.,100.])
    expected=min(report['trials'],key=lambda row:row['huber_loss'])['alpha']
    assert model['alpha']==expected
    assert label_scale([.01,.02,np.nan])['unit'].startswith('decimal')

@pytest.mark.parametrize('channels,outputs,groups,kernel,padding',[(128,128,128,7,3),(128,256,16,1,0),(256,128,8,1,0)])
def test_cpu_conv_forward_and_gradient_equivalence(channels,outputs,groups,kernel,padding):
    import torch
    from torch import nn
    from src.models.cpu_conv import ShortSequenceConv1d
    torch.manual_seed(3)
    native=nn.Conv1d(channels,outputs,kernel,groups=groups,padding=padding).double()
    fast=ShortSequenceConv1d(channels,outputs,kernel,groups=groups,padding=padding).double()
    fast.load_state_dict(native.state_dict())
    x=torch.randn(2,channels,12,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    out1,out2=native(x),fast(y)
    torch.testing.assert_close(out1,out2,rtol=1e-12,atol=1e-12)
    target=torch.randn_like(out1)
    (out1*target).sum().backward();(out2*target).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-12,atol=1e-12)
    for a,b in zip(native.parameters(),fast.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
