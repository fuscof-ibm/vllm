# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Targeted unit tests for VLLM_TRITON_FORCE_FIRST_CONFIG.

These tests exercise only the patched `Autotuner.run` logic installed by
`vllm.env_override._disable_triton_autotuner`. The wrapped kernel is a
plain callable so the tests run on CPU-only hosts (no GPU, no actual
kernel launch) as long as the `triton` package is importable.
"""

from types import SimpleNamespace

import pytest

from vllm.triton_utils import HAS_TRITON, triton

if not HAS_TRITON:
    pytest.skip("triton not available", allow_module_level=True)

from vllm.env_override import _disable_triton_autotuner  # noqa: E402

OutOfResources = triton.runtime.errors.OutOfResources


@pytest.fixture
def patched_autotuner(monkeypatch: pytest.MonkeyPatch):
    """Install the VLLM_TRITON_FORCE_FIRST_CONFIG patch and restore after."""
    Autotuner = triton.runtime.autotuner.Autotuner
    original_run = Autotuner.run
    monkeypatch.setenv("VLLM_TRITON_FORCE_FIRST_CONFIG", "1")
    _disable_triton_autotuner()
    yield Autotuner
    Autotuner.run = original_run


def _make_fake_self(configs, fn):
    """Minimal stand-in for an Autotuner instance."""
    return SimpleNamespace(
        configs=configs,
        keys=[],
        arg_names=[],
        base_fn=fn,
        fn=fn,
        best_config=None,
    )


def test_skips_invalid_first_config_and_caches_second(patched_autotuner):
    bad = triton.Config({"BLOCK": 1024})
    good = triton.Config({"BLOCK": 64})
    calls = []

    def fake_fn(*args, **kwargs):
        calls.append(kwargs["BLOCK"])
        if kwargs["BLOCK"] == 1024:
            raise OutOfResources(required=99999, limit=1, name="shared memory")
        return "ok"

    fake_self = _make_fake_self([bad, good], fake_fn)

    # First call: walks past the invalid config, picks the second.
    assert patched_autotuner.run(fake_self) == "ok"
    assert calls == [1024, 64]
    assert fake_self.best_config is good

    # Second call: cached index is reused, invalid config is NOT retried.
    calls.clear()
    assert patched_autotuner.run(fake_self) == "ok"
    assert calls == [64]


def test_empty_configs_falls_back_to_direct_call(patched_autotuner):
    def fake_fn(*args, **kwargs):
        return "direct"

    fake_self = _make_fake_self([], fake_fn)
    assert patched_autotuner.run(fake_self) == "direct"


def test_all_configs_invalid_raises_runtime_error(patched_autotuner):
    cfgs = [triton.Config({"BLOCK": 1024}), triton.Config({"BLOCK": 2048})]

    def always_oor(*args, **kwargs):
        raise OutOfResources(required=99999, limit=1, name="shared memory")

    fake_self = _make_fake_self(cfgs, always_oor)
    with pytest.raises(RuntimeError, match="no valid config"):
        patched_autotuner.run(fake_self)
