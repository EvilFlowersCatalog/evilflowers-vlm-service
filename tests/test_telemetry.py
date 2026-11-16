from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI

from evilflowers_vlm import telemetry
from evilflowers_vlm.__main__ import parse_args


def _record(msg: str, exc: bool = False) -> logging.LogRecord:
    exc_info = None
    if exc:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
    return logging.LogRecord("evilflowers_vlm.app", logging.WARNING, __file__, 1, msg, (), exc_info)


def test_json_log_line_has_level_and_no_trace_outside_a_span() -> None:
    line = json.loads(telemetry.JsonFormatter().format(_record("hello %s")))
    assert line["level"] == "warning" and line["message"] == "hello %s"
    assert line["logger"] == "evilflowers_vlm.app" and "trace_id" not in line


def test_json_log_line_carries_exception() -> None:
    line = json.loads(telemetry.JsonFormatter().format(_record("failed", exc=True)))
    assert "ValueError: boom" in line["exc_info"]


@pytest.mark.parametrize(("env", "on"), [
    ({}, False),
    ({"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, False),
    ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://10.0.0.1:4318"}, True),
    ({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://x:4318/v1/traces"}, True),
    ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318", "OTEL_SDK_DISABLED": "true"}, False),
])
def test_otlp_configured(env: dict[str, str], on: bool) -> None:
    assert telemetry.otlp_configured(env) is on


def test_tracing_off_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert telemetry.setup_tracing(FastAPI()) is None


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_MODEL_DIR", "/m/x")
    monkeypatch.setenv("VLM_DEVICE", "cpu")
    monkeypatch.setenv("VLM_THREADS", "6")
    monkeypatch.setenv("VLM_MAX_QUEUE", "4")
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    a = parse_args([])
    assert (str(a.model_dir), a.device, a.threads, a.max_queue, a.log_format) == ("/m/x", "cpu", 6, 4, "json")
    assert a.use_cache is True
