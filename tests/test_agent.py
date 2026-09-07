import json
from unittest.mock import Mock

import numpy as np
import pytest
import requests

from core.graph_edit_agent import LLMGraphAgent, _is_transient


def test_key_required():
    with pytest.raises(ValueError):
        LLMGraphAgent("", ["A", "B"], "context")


@pytest.mark.parametrize(
    "status,retry",
    [(400, False), (401, False), (403, False), (429, True), (500, True), (503, True)],
)
def test_only_transient_http_statuses_retry(status, retry):
    response = requests.Response()
    response.status_code = status
    assert _is_transient(requests.HTTPError(response=response)) is retry


def test_connection_and_timeout_are_retryable():
    assert _is_transient(requests.Timeout())
    assert _is_transient(requests.ConnectionError())
    assert not _is_transient(ValueError())


def test_transport_has_timeout(monkeypatch):
    response = Mock()
    response.json.return_value = {"ok": True}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    agent = LLMGraphAgent("fake-key", ["A", "B"], "context")
    assert agent._call_llm_api({"test": True}) == {"ok": True}
    assert post.call_args.kwargs["timeout"] == (10, 90)
    response.raise_for_status.assert_called_once()


def test_schema_hallucination_and_noop_filters(monkeypatch):
    edits = [
        None,
        "bad",
        {"action": [], "source": "A", "target": "B"},
        {"action": "dance", "source": "A", "target": "B"},
        {"action": "add", "source": "unknown", "target": "B"},
        {"action": "add", "source": "A", "target": "A"},
        {"action": "delete", "source": "B", "target": "A"},
        {"action": "add", "source": "A", "target": "B"},
        {"action": "add", "source": "A", "target": "B"},
    ]
    agent = LLMGraphAgent("fake-key", ["A", "B"], "context", model="test-model")
    fake = Mock(return_value={"choices": [{"message": {"content": json.dumps({"edits": edits})}}]})
    monkeypatch.setattr(agent, "_call_llm_api", fake)
    result = agent.propose_edits([], [], "", np.ones((2, 2)))
    assert result == [{"action": "add", "source": "A", "target": "B"}]
    assert fake.call_args.args[0]["model"] == "test-model"


@pytest.mark.parametrize("content", ["[]", '{"edits": null}', '{"edits": {}}', "not-json"])
def test_malformed_response_is_rejected(monkeypatch, content):
    agent = LLMGraphAgent("fake-key", ["A", "B"], "context")
    monkeypatch.setattr(
        agent, "_call_llm_api", lambda payload: {"choices": [{"message": {"content": content}}]}
    )
    assert agent.propose_edits([], [], "", np.ones((2, 2))) == []
