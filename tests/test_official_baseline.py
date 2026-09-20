import numpy as np
import pandas as pd
import pytest
from src.data.build_samples import build_baseline_panel
from src.models.official_linear_baseline import dense_monthly_transform, fit_official_year, official_grids

FACTORS = [f'f{i}' for i in range(147)]
SMALL_GRID = {'ols':[None], 'lasso':[.01], 'ridge':[1.0], 'en':[.01]}


def example():
    rows=[]
    for eom in ['2018-11-30','2018-12-31','2020-12-31']:
        for i in range(6):
            rows.append({'permno':i+1,'date':eom,'eom':eom,'ret_exc_lead1m':.02*(i-2),
                         **{f:float(i) for f in FACTORS}})
    return build_baseline_panel(pd.DataFrame(rows))


def test_dense_ties_differ_from_average_and_degenerate_safe():
    panel=example().iloc[:6].copy()
    panel[FACTORS[0]]=[1,1,2,2,3,np.nan]
    panel[FACTORS[1]]=np.nan
    panel[FACTORS[2]]=7
    x,_=dense_monthly_transform(panel,FACTORS)
    np.testing.assert_array_equal(x[FACTORS[0]],[-1,-1,0,0,1,0])
    assert (x[FACTORS[1:3]]==0).all().all()
    single,_=dense_monthly_transform(panel.iloc[:1],FACTORS)
    assert (single==0).all().all()


def test_missing_future_labels_leave_predictions_and_scaler_unchanged():
    panel=example(); x,_=dense_monthly_transform(panel,FACTORS)
    before,_,params=fit_official_year(panel,x,FACTORS,2021,SMALL_GRID)
    panel.loc[12:,'ret_exc_lead1m']=np.nan
    after,_,params2=fit_official_year(panel,x,FACTORS,2021,SMALL_GRID)
    np.testing.assert_array_equal(before[list(SMALL_GRID)],after[list(SMALL_GRID)])
    assert params==params2 and len(after)==6


def test_validation_data_not_fitted_by_scaler_or_coefficients():
    panel=example(); x,_=dense_monthly_transform(panel,FACTORS)
    _,_,params=fit_official_year(panel,x,FACTORS,2021,SMALL_GRID)
    x.loc[6:11]=999
    panel.loc[6:11,'ret_exc_lead1m']=100
    _,_,after=fit_official_year(panel,x,FACTORS,2021,SMALL_GRID)
    assert params==after
    np.testing.assert_allclose(params['scaler_mean'],x.iloc[:6].mean(axis=0))
    np.testing.assert_allclose(params['scaler_scale'],x.iloc[:6].std(axis=0,ddof=0))


def test_grid_sizes_and_official_ridge_half_multiplier():
    grids=official_grids()
    assert [len(grids[k]) for k in ['ols','lasso','ridge','en']]==[1,81,91,81]
    assert grids['ridge'][0]==pytest.approx(.05)
    assert grids['ridge'][-1]==pytest.approx(5e7)


def test_future_cross_section_has_no_effect_on_history():
    panel=example(); before,_=dense_monthly_transform(panel,FACTORS)
    panel.loc[12:,FACTORS]=-500
    after,_=dense_monthly_transform(panel,FACTORS)
    pd.testing.assert_frame_equal(before.iloc[:12],after.iloc[:12])
