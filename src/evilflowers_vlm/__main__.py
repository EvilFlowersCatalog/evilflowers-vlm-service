"""``python -m evilflowers_vlm``: serve PaddleOCR-VL on ``POST /v1/formulas``.

Every flag has an environment default (``VLM_*``, ``LOG_FORMAT``) so the k8s manifest
stays short. ``VLM_THREADS`` must equal the container's CPU limit: torch sizes its
thread pool from the host's cores, not the cgroup limit.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    return None if v is None or v == "" else v.lower() in {"1", "true", "yes"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="evilflowers_vlm", description=__doc__)
    ap.add_argument("--host", default=os.environ.get("VLM_HOST", "0.0.0.0"))  # noqa: S104
    ap.add_argument("--port", type=int, default=int(os.environ.get("VLM_PORT", "8000")))
    ap.add_argument("--model-dir", type=Path, default=Path(os.environ.get("VLM_MODEL_DIR", "/models")))
    ap.add_argument("--device", default=os.environ.get("VLM_DEVICE", "cpu"))
    ap.add_argument("--dtype", default=os.environ.get("VLM_DTYPE", "auto"),
                    help="auto (float32 on CPU) | float32 | bfloat16")
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("VLM_THREADS", os.environ.get("OMP_NUM_THREADS", "0"))),
                    help="torch intra-op threads; 0 = torch default (HOST cores, not the cgroup limit)")
    ap.add_argument("--max-queue", type=int, default=int(os.environ.get("VLM_MAX_QUEUE", "2")),
                    help="requests allowed to wait behind the running one before 503")
    ap.add_argument("--log-format", choices=["text", "json"],
                    default=os.environ.get("LOG_FORMAT", "text").lower())
    ap.add_argument("--log-level", default=os.environ.get("VLM_LOG_LEVEL", "INFO"))
    # The checkpoint's generation_config says use_cache=false, which re-runs the whole
    # sequence per token. With the KV cache: identical text, ~3.6x faster on CPU.
    cache = ap.add_mutually_exclusive_group()
    cache.add_argument("--kv-cache", dest="use_cache", action="store_true")
    cache.add_argument("--no-kv-cache", dest="use_cache", action="store_false")
    env_cache = _env_bool("VLM_KV_CACHE")
    ap.set_defaults(use_cache=True if env_cache is None else env_cache)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)

    from . import telemetry

    telemetry.setup_logging(a.log_level, a.log_format)
    log = logging.getLogger("evilflowers_vlm")
    if a.threads <= 0:
        log.warning("VLM_THREADS is not set: torch will size its pool from the host's cores")

    import uvicorn

    from .app import create_app
    from .engine import PaddleEngine

    engine = PaddleEngine(a.model_dir, device=a.device, dtype=a.dtype, threads=a.threads,
                          use_cache=a.use_cache)
    app = create_app(engine, max_queue=a.max_queue)
    provider = telemetry.setup_tracing(app)
    try:
        # log_config=None: keep our handlers, so uvicorn's lines are JSON too
        uvicorn.run(app, host=a.host, port=a.port, log_config=None, log_level=a.log_level.lower())
    finally:
        if provider is not None:
            provider.shutdown()


if __name__ == "__main__":
    main()
