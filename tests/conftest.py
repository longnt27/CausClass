"""All automated tests are offline: deny socket connections globally."""

import socket

import pytest
import torch


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch, tmp_path):
    def deny_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in tests; mock the transport")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from core.models import aerca

    monkeypatch.setattr(aerca, "OUTPUT_DIR", tmp_path / "model-output")
    torch.set_num_threads(1)
