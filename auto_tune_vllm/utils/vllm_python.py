"""Interpreter used to run vLLM trial servers.

The tuner only talks to vLLM through subprocesses, so vLLM may live in a different
Python environment than the tuner itself (e.g. the system Python of a vLLM image,
while the tuner runs from an isolated venv). Set VLLM_PYTHON to that interpreter.
"""

import os
import subprocess

VLLM_PYTHON_ENV = "VLLM_PYTHON"
DEFAULT_VLLM_PYTHON = "python3"
# Module behind the `vllm` console script; `python -m vllm` fails (no __main__.py).
VLLM_CLI_MODULE = "vllm.entrypoints.cli.main"


def get_vllm_python() -> str:
    """Return the interpreter that runs vLLM, defaulting to python3 on PATH."""
    return os.environ.get(VLLM_PYTHON_ENV) or DEFAULT_VLLM_PYTHON


def is_vllm_importable(python_bin: str) -> bool:
    """Check that vLLM can be imported by the given interpreter."""
    try:
        result = subprocess.run(
            [python_bin, "-c", "import vllm"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
