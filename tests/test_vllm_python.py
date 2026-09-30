"""Running vLLM trials with an interpreter outside the tuner's environment."""

import io
import logging
import sys
from types import SimpleNamespace

import pytest

from auto_tune_vllm.execution import trial_controller
from auto_tune_vllm.execution.backends import RayExecutionBackend
from auto_tune_vllm.execution.trial_controller import LocalTrialController
from auto_tune_vllm.utils.vllm_cli_parser import VLLMCLIParser
from auto_tune_vllm.utils.vllm_python import (
    VLLM_PYTHON_ENV,
    get_vllm_python,
    is_vllm_importable,
)


@pytest.fixture(autouse=True)
def _no_vllm_python(monkeypatch):
    monkeypatch.delenv(VLLM_PYTHON_ENV, raising=False)


def test_defaults_to_python3_on_path():
    assert get_vllm_python() == "python3"


def test_empty_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(VLLM_PYTHON_ENV, "")
    assert get_vllm_python() == "python3"


def test_override_selects_interpreter(monkeypatch):
    monkeypatch.setenv(VLLM_PYTHON_ENV, "/usr/bin/python3.12")
    assert get_vllm_python() == "/usr/bin/python3.12"


def test_vllm_importable_when_interpreter_can_import_it(monkeypatch, tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert is_vllm_importable(sys.executable)


def test_vllm_not_importable_from_missing_interpreter():
    assert not is_vllm_importable("/nonexistent/bin/python")


def _start_server(monkeypatch) -> list[str]:
    launched = {}

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return SimpleNamespace(stdout=io.StringIO(""), pid=1234)

    monkeypatch.setattr(trial_controller.subprocess, "Popen", fake_popen)
    controller = LocalTrialController()
    monkeypatch.setattr(controller, "_get_trial_logger", logging.getLogger)
    monkeypatch.setattr(controller, "_log_python_environment", lambda _: None)
    trial = SimpleNamespace(
        benchmark_config=SimpleNamespace(model="/model"),
        vllm_args=["--max-num-seqs", "64"],
        environment_vars={},
    )
    controller._start_vllm_server(trial)
    return launched["cmd"]


def test_server_launches_with_python3_by_default(monkeypatch):
    cmd = _start_server(monkeypatch)
    assert cmd[:3] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert cmd[-2:] == ["--max-num-seqs", "64"]


def test_server_launches_with_vllm_python(monkeypatch):
    monkeypatch.setenv(VLLM_PYTHON_ENV, "/usr/bin/python3.12")
    assert _start_server(monkeypatch)[0] == "/usr/bin/python3.12"


def _backend_without_ray() -> RayExecutionBackend:
    backend = object.__new__(RayExecutionBackend)
    backend.python_executable = "/venv/bin/python"
    backend.venv_path = None
    backend.conda_env = None
    return backend


def test_ray_workers_receive_vllm_python(monkeypatch):
    monkeypatch.setenv(VLLM_PYTHON_ENV, "/usr/bin/python3.12")
    runtime_env = _backend_without_ray()._build_runtime_env()
    assert runtime_env["env_vars"] == {VLLM_PYTHON_ENV: "/usr/bin/python3.12"}
    assert runtime_env["python"] == "/venv/bin/python"


def test_ray_runtime_env_unchanged_without_vllm_python():
    assert "env_vars" not in _backend_without_ray()._build_runtime_env()


def test_cli_parser_prefers_vllm_python_over_venv(monkeypatch):
    monkeypatch.setenv(VLLM_PYTHON_ENV, "/usr/bin/python3.12")
    assert VLLMCLIParser(venv_path="/venv")._python_bin() == "/usr/bin/python3.12"


def test_cli_parser_uses_venv_without_vllm_python():
    assert VLLMCLIParser(venv_path="/venv")._python_bin() == "/venv/bin/python"
