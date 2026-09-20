"""Verify the submitted entrypoint's stage order and failure boundary."""
from types import SimpleNamespace
import pytest
import MAIN
import src.portfolio.reporting as reporting


@pytest.mark.parametrize('kind', ['quant', 'multimodal'])
def test_full_workflow_order(monkeypatch, kind):
    calls = []
    for name in ('audit', 'prepare', 'embed', 'backtest'):
        monkeypatch.setattr(MAIN, name, lambda *args, name=name: calls.append(name) or {})
    for name in ('train', 'predict'):
        monkeypatch.setattr(MAIN, name, lambda config, kind, year, name=name: calls.append((name, year)))
    monkeypatch.setattr(reporting, 'build_report', lambda path: calls.append('report') or {})
    args = SimpleNamespace(kind=kind, start=2025, end=2026, output_dir='unused', pptx=False)
    result = MAIN.run_pipeline({}, args)
    expected = ['audit', 'prepare'] + (['embed'] if kind == 'multimodal' else [])
    expected += [('train', 2025), ('predict', 2025), ('train', 2026), ('predict', 2026), 'backtest', 'report']
    assert calls == expected
    assert result == {'backtest': {}, 'report': {}}


def test_failed_audit_stops_workflow(monkeypatch):
    def fail(config):
        raise ValueError('invalid input')
    monkeypatch.setattr(MAIN, 'audit', fail)
    monkeypatch.setattr(MAIN, 'prepare', lambda config: pytest.fail('must stop after failed audit'))
    with pytest.raises(ValueError, match='invalid input'):
        MAIN.run_pipeline({}, SimpleNamespace())
