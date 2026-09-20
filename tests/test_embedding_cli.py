import json
import sys
from filelock import Timeout
import pytest
from scripts import precompute_embeddings as cli


def setup(monkeypatch,tmp_path,result,args=()):
    monkeypatch.setattr(sys,'argv',['precompute_embeddings',*args])
    monkeypatch.setattr(cli,'read_config',lambda _:({'source_path':'source','cache_dir':str(tmp_path),'model_cache_dir':'model'},object()))
    monkeypatch.setattr(cli,'run_pipeline',lambda *a,**kw:result)


@pytest.mark.parametrize('show_monthly',[False,True])
def test_terminal_summary_and_monthly_option(monkeypatch,tmp_path,capsys,show_monthly):
    report={'source_rows':1,'by_filing_month':{'2020-12':{'success':1}}}
    setup(monkeypatch,tmp_path,report,['--show-monthly'] if show_monthly else [])
    cli.main()
    captured=capsys.readouterr()
    data=json.loads(captured.out)
    assert ('by_filing_month' in data)==show_monthly
    assert report['by_filing_month']=={'2020-12':{'success':1}}
    assert str(tmp_path/'coverage_report.json') in captured.err


@pytest.mark.parametrize('result',[None,[],RuntimeError('returned exception')])
def test_invalid_report_is_explicit_error(monkeypatch,tmp_path,capsys,result):
    setup(monkeypatch,tmp_path,result)
    with pytest.raises(SystemExit) as exc:cli.main()
    assert exc.value.code==1
    captured=capsys.readouterr()
    assert 'expected a report dictionary' in captured.err
    assert not captured.out


@pytest.mark.parametrize('failure',[OSError('missing source'),ValueError('identity mismatch'),RuntimeError('device fault'),Timeout('cache.lock')])
def test_pipeline_failure_has_nonzero_exit(monkeypatch,tmp_path,capsys,failure):
    setup(monkeypatch,tmp_path,None)
    def fail(*a,**kw):raise failure
    monkeypatch.setattr(cli,'run_pipeline',fail)
    with pytest.raises(SystemExit) as exc:cli.main()
    assert exc.value.code==1
    captured=capsys.readouterr()
    assert 'Encoding failed' in captured.err and '--debug' in captured.err
    assert not captured.out


def test_debug_preserves_original_exception(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,None,['--debug'])
    failure=RuntimeError('original device fault')
    def fail(*a,**kw):raise failure
    monkeypatch.setattr(cli,'run_pipeline',fail)
    with pytest.raises(RuntimeError) as exc:cli.main()
    assert exc.value is failure
