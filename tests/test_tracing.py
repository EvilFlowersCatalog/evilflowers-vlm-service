"""Tracing on: a SERVER span per request, a child ``vlm.batch`` span, probes untraced.

Needs the ``otel`` extra (``uv sync --extra otel``). It sets the global tracer provider,
which OpenTelemetry allows once per process, so this module holds the only such test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.instrumentation.fastapi")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from evilflowers_vlm import telemetry
from evilflowers_vlm.app import create_app

from .test_app import StubEngine, _png


def test_spans_when_an_endpoint_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "vlm")
    monkeypatch.setattr("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
                        lambda: exporter)
    app = create_app(StubEngine(), load_in_background=False)
    provider = telemetry.setup_tracing(app)
    assert provider is not None
    with TestClient(app) as client:
        client.get("/readyz")
        client.get("/metrics")
        resp = client.post("/v1/formulas", json={"items": [
            {"key": "a", "image": _png(20, 20)}, {"key": "b", "image": _png(99, 99)}]})
    assert resp.status_code == 200
    provider.force_flush()
    spans = exporter.get_finished_spans()
    server = [s for s in spans if s.kind == SpanKind.SERVER]
    assert [s.name for s in server] == ["POST /v1/formulas"]          # probes and /metrics excluded
    batch = next(s for s in spans if s.name == "vlm.batch")
    assert batch.parent is not None and batch.parent.span_id == server[0].context.span_id
    assert batch.attributes is not None
    assert (batch.attributes["vlm.items"], batch.attributes["vlm.items.failed"]) == (2, 1)
    assert server[0].resource.attributes["service.name"] == "vlm"
    provider.shutdown()
