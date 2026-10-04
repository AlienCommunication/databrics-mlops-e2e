"""Failure-path tests: a rejected or broken online candidate must not become Champion."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import workflow as module

def fixture_workflow(monkeypatch):
    flow=module.Workflow.__new__(module.Workflow)
    flow.model_name='catalog.schema.model'; flow.endpoint='endpoint'
    flow.catalog='catalog';flow.schema='schema';flow.release='test'
    flow.client=MagicMock();flow.client.get_model_version.return_value.tags={'validation':'passed'}
    flow.get=lambda task,key: {'version':'2','previous':'1'}[key]
    flow.alias_version=lambda alias:'1'
    flow.merge=MagicMock()
    workspace=MagicMock()
    monkeypatch.setattr(module,'WorkspaceClient',lambda:workspace)
    return flow,workspace

def test_rejected_model_cannot_reach_serving(monkeypatch):
    flow,w=fixture_workflow(monkeypatch)
    flow.client.get_model_version.return_value.tags={'validation':'failed'}
    with pytest.raises(ValueError,match='validation gate'):flow.deploy()
    w.serving_endpoints.update_config_and_wait.assert_not_called()
    flow.client.set_registered_model_alias.assert_not_called()

def test_smoke_failure_restores_old_online_version(monkeypatch):
    flow,w=fixture_workflow(monkeypatch)
    w.serving_endpoints.query.return_value=SimpleNamespace(predictions=[999,999])
    monkeypatch.setattr(module,'load_model',lambda uri:SimpleNamespace(predict=lambda frame:[100,100]))
    with pytest.raises(AssertionError):flow.deploy()
    calls=w.serving_endpoints.update_config_and_wait.call_args_list
    assert calls[0].kwargs['served_entities'][0].entity_version=='2'
    assert calls[-1].kwargs['served_entities'][0].entity_version=='1'
    flow.client.set_registered_model_alias.assert_not_called()

def test_success_pins_version_before_alias(monkeypatch):
    flow,w=fixture_workflow(monkeypatch)
    w.serving_endpoints.query.return_value=SimpleNamespace(predictions=[100,100])
    monkeypatch.setattr(module,'load_model',lambda uri:SimpleNamespace(predict=lambda frame:[100,100]))
    flow.deploy()
    assert w.serving_endpoints.update_config_and_wait.call_args.kwargs['served_entities'][0].entity_version=='2'
    flow.client.set_registered_model_alias.assert_any_call(flow.model_name,'Champion','2')

def test_champion_change_requires_revalidation(monkeypatch):
    flow,w=fixture_workflow(monkeypatch)
    flow.alias_version=lambda alias:'3'
    with pytest.raises(ValueError,match='changed after validation'):flow.deploy()
    w.serving_endpoints.update_config_and_wait.assert_not_called()
