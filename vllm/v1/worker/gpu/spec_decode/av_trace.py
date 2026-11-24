# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research tracing of adaptive verification's draft-budget decisions.

Enabled by VLLM_AV_TRACE_DIR; see PR_adaptive_verification_perf/B_instrumentation.md.
Never synchronizes with the device while serving: device data reaches the host
through async copies and events that are only polled with query().
"""

import atexit
import json
import os
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import pybase64 as base64
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_COPY_SLOTS = 16
_MAX_PENDING_STEPS = 64
_FLUSH_SECONDS = 0.5
# GPU intervals reported per step: (field, start event, end event).
_INTERVALS = (
    ("prep_ms", "step_start", "forward_start"),
    ("forward_ms", "forward_start", "forward_end"),
    ("sample_ms", "forward_end", "sample_end"),
    ("draft_ms", "drafter_start", "drafter_end"),
    ("kernel_ms", "kernel_start", "kernel_end"),
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.float32:
            # Exact and compact: the replay must reproduce B bit for bit.
            raw = np.ascontiguousarray(obj).tobytes()
            return {"f4": base64.b64encode(raw).decode(), "shape": list(obj.shape)}
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def decode_f4(value: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(value["f4"])
    return np.frombuffer(raw, dtype=np.float32).reshape(value["shape"])


class AVTracer:
    """Collects trace records of one DP rank and writes them as JSONL."""

    @staticmethod
    def maybe_create(meta: dict[str, Any], tp_rank: int) -> "AVTracer | None":
        out_dir = os.environ.get("VLLM_AV_TRACE_DIR")
        if not out_dir or tp_rank != 0:
            return None
        return AVTracer(out_dir, meta)

    def __init__(self, out_dir: str, meta: dict[str, Any]):
        os.makedirs(out_dir, exist_ok=True)
        self.meta = dict(meta)
        self.dp_rank = int(self.meta.get("dp_rank") or 0)
        self.path = os.path.join(
            out_dir, f"av_trace_dp{self.dp_rank}_{os.getpid()}.jsonl"
        )
        self.step = 0
        self.events_enabled = os.environ.get("VLLM_AV_TRACE_EVENTS", "1") != "0"
        window = os.environ.get("VLLM_AV_TRACE_WINDOW")
        self._window = tuple(int(x) for x in window.split(",")) if window else None
        self.drops = {"copy": 0, "timing": 0}

        # Async copies: pinned host slots, allocated on first use.
        self._copy_bufs: list[torch.Tensor] = []
        self._copy_events: list[torch.Event] = []
        self._copy_pending: list[dict[str, Any] | None] = [None] * _COPY_SLOTS
        self._copy_next = 0

        # Timing events of the current step, and of steps not resolved yet.
        self._event_pool: list[torch.Event] = []
        self._cur_events: dict[str, torch.Event] = {}
        self._cur_info: dict[str, Any] = {}
        self._pending_steps: deque[
            tuple[int, dict[str, torch.Event], dict[str, Any]]
        ] = deque()

        self._queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()
        self._closed = False
        self._writer = threading.Thread(
            target=self._write_loop, name="av-trace-writer", daemon=True
        )
        self._writer.start()
        atexit.register(self.close)
        logger.info("Adaptive verification trace: %s", self.path)

    def active(self, step: int | None = None) -> bool:
        if self._window is None:
            return True
        on, off = self._window
        return (self.step if step is None else step) % (on + off) < on

    def begin_step(self) -> None:
        if self._cur_events:
            self._pending_steps.append((self.step, self._cur_events, self._cur_info))
            if len(self._pending_steps) > _MAX_PENDING_STEPS:
                _, events, _ = self._pending_steps.popleft()
                self._event_pool.extend(events.values())
                self.drops["timing"] += 1
        self._cur_events, self._cur_info = {}, {}
        self.step += 1
        self.poll()

    def log(
        self, kind: str, step: int | None = None, force: bool = False, **fields: Any
    ) -> None:
        step = self.step if step is None else step
        if force or self.active(step):
            self._queue.put(
                {"type": kind, "step": step, "dp_rank": self.dp_rank, **fields}
            )

    def set_step_info(self, **info: Any) -> None:
        if self.active():
            self._cur_info.update(info)

    def event(self, name: str) -> None:
        """Record a timing event for the current step on the current stream."""
        if not self.events_enabled or not self.active():
            return
        ev = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        ev.record()
        self._cur_events[name] = ev

    def copy_async(
        self, kind: str, field: str, src: torch.Tensor, rows: np.ndarray, **fields: Any
    ) -> None:
        """Copy `src` to pinned memory on the current stream, then emit
        `src[rows]` under `field` once the copy has landed. Drops the record
        rather than block when every slot is still in flight."""
        if not self.active():
            return
        if not self._copy_bufs:
            self._copy_bufs = [
                torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
                for _ in range(_COPY_SLOTS)
            ]
            self._copy_events = [torch.Event() for _ in range(_COPY_SLOTS)]
        slot = self._copy_next
        if self._copy_pending[slot] is not None:
            self._poll_copies()
            if self._copy_pending[slot] is not None:
                self.drops["copy"] += 1
                return
        self._copy_bufs[slot].copy_(src, non_blocking=True)
        self._copy_events[slot].record()
        self._copy_pending[slot] = {
            "type": kind,
            "step": self.step,
            "dp_rank": self.dp_rank,
            "_field": field,
            "_rows": np.array(rows, copy=True),
            **fields,
        }
        self._copy_next = (slot + 1) % _COPY_SLOTS

    def poll(self) -> None:
        self._poll_copies()
        self._poll_timing()

    def _poll_copies(self) -> None:
        for slot, record in enumerate(self._copy_pending):
            if record is None or not self._copy_events[slot].query():
                continue
            rows = record.pop("_rows")
            record[record.pop("_field")] = self._copy_bufs[slot].numpy()[rows]
            self._queue.put(record)
            self._copy_pending[slot] = None

    def _poll_timing(self) -> None:
        while self._pending_steps:
            step, events, info = self._pending_steps[0]
            # One stream, so the last event recorded completes last.
            last = next(reversed(events.values()))
            if not last.query():
                return
            self._pending_steps.popleft()
            record: dict[str, Any] = {
                "type": "timing",
                "step": step,
                "dp_rank": self.dp_rank,
                **info,
            }
            for name, start, end in _INTERVALS:
                if start in events and end in events:
                    record[name] = round(events[start].elapsed_time(events[end]), 4)
            first = next(iter(events.values()))
            record["step_ms"] = round(first.elapsed_time(last), 4)
            self._queue.put(record)
            self._event_pool.extend(events.values())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._cur_events:
                self._pending_steps.append(
                    (self.step, self._cur_events, self._cur_info)
                )
                self._cur_events = {}
            # Shutdown only: resolve what is still in flight.
            torch.accelerator.synchronize()
            self.poll()
        except Exception:  # noqa: BLE001 - the device may already be gone
            pass
        self._queue.put(
            {
                "type": "summary",
                "step": self.step,
                "dp_rank": self.dp_rank,
                "drops": dict(self.drops),
            }
        )
        self._queue.put(None)
        self._writer.join(timeout=10)

    def _write_loop(self) -> None:
        with open(self.path, "w", buffering=1 << 20) as f:
            last_flush = time.monotonic()
            while True:
                try:
                    record = self._queue.get(timeout=_FLUSH_SECONDS)
                except queue.Empty:
                    record = {}
                if record is None:
                    f.flush()
                    return
                if record:
                    f.write(
                        json.dumps(record, default=_jsonable, separators=(",", ":"))
                    )
                    f.write("\n")
                if time.monotonic() - last_flush >= _FLUSH_SECONDS:
                    f.flush()
                    last_flush = time.monotonic()
