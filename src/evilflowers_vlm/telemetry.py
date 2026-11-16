"""Telemetry per the knowledge note ``Deploy-a-Service`` (logs, metrics, spans).

* Logs: ``LOG_FORMAT=json`` gives one JSON object per line with ``level`` and, while a span
  is active, ``trace_id`` / ``span_id`` (Loki's *View trace* link needs ``trace_id``).
* Metrics: Prometheus ``vlm_*`` series on ``GET /metrics`` (pull, scraped by Alloy when the
  pod carries ``k8s.grafana.com/scrape: "true"``). Nothing is pushed.
* Spans: OTLP/HTTP through the standard ``OTEL_*`` variables. Without
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or ``…_TRACES_ENDPOINT``), or without the ``otel`` extra,
  no SDK is installed and every span is the API's no-op.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("evilflowers_vlm")

# Labels from closed sets only: task ∈ {formula, ocr}, outcome ∈ {ok, error},
# reason ∈ {loading, busy}.
ITEMS = Counter("vlm_items", "Crops processed, by task and outcome.", ["task", "outcome"])
ITEM_SECONDS = Histogram(
    "vlm_item_seconds", "Wall time of one crop: decode, normalize and inference.", ["task"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300))
QUEUE_REJECTIONS = Counter(
    "vlm_queue_rejections", "Requests answered 503 + Retry-After, by reason.", ["reason"])

_TRACES_EXCLUDED = "healthz,readyz,metrics"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, trace_id/span_id, exc_info."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            out["trace_id"] = format(ctx.trace_id, "032x")
            out["span_id"] = format(ctx.span_id, "016x")
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Root logging on stderr; ``fmt="json"`` for Loki. uvicorn must run with ``log_config=None``."""
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def otlp_configured(env: dict[str, str] | None = None) -> bool:
    env = dict(os.environ) if env is None else env
    if env.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return bool(env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))


def setup_tracing(app: FastAPI) -> Any | None:
    """Install the SDK and a SERVER span per request; return the provider (to shut down) or None."""
    if not otlp_configured():
        logger.info("tracing off: no OTLP endpoint configured")
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("tracing off: OTLP endpoint set but the `otel` extra is not installed")
        return None
    # Resource.create() reads OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, excluded_urls=_TRACES_EXCLUDED)
    logger.info("tracing on: OTLP/HTTP export")
    return provider
