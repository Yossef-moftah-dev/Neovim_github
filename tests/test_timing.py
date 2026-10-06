"""Tests for timing decorator."""

from __future__ import annotations

import asyncio
import time

from prodml.timing import timed


def test_sync_timed_decorator() -> None:
    @timed
    def sample_func(x: int) -> int:
        time.sleep(0.01)
        return x * 2

    res = sample_func(5)
    assert res == 10
    assert hasattr(sample_func, "last_duration_ms")
    assert sample_func.last_duration_ms >= 5.0  # at least ~5ms


def test_async_timed_decorator() -> None:
    @timed
    async def sample_async_func(msg: str) -> str:
        await asyncio.sleep(0.01)
        return f"done: {msg}"

    res = asyncio.run(sample_async_func("test"))
    assert res == "done: test"
    assert hasattr(sample_async_func, "last_duration_ms")
    assert sample_async_func.last_duration_ms >= 5.0
