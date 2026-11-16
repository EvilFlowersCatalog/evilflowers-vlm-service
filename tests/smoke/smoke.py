"""Smoke test of the built image: real model, one rendered formula crop.

Run inside the container, read-only and offline (CI and README):

    docker run --rm --read-only --tmpfs /tmp --network none --user 65534:65534 \
      -v "$PWD/tests/smoke:/smoke:ro" evilflowers-vlm-service python /smoke/smoke.py --serve

``--serve`` starts ``python -m evilflowers_vlm`` itself (with ``--network none`` nothing
outside the container can reach it); without it, ``--url`` points at a running server.
Stdlib + Pillow only. Exit 0 only if ``/readyz`` turns 200 and the crop comes back as text.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont

FORMULA = "x^2 + y^2 = z^2"


def _crop() -> str:
    im = Image.new("RGB", (420, 90), "white")
    ImageDraw.Draw(im).text((20, 15), FORMULA, fill="black", font=ImageFont.load_default(size=48))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - fixed local URL
            return int(r.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=600)
    a = ap.parse_args()

    server = None
    if a.serve:
        env = {**os.environ, "VLM_THREADS": os.environ.get("VLM_THREADS", str(os.cpu_count() or 1))}
        server = subprocess.Popen([sys.executable, "-m", "evilflowers_vlm"], env=env)
    try:
        t0 = time.monotonic()
        while (status := _get(a.url + "/readyz")) != 200:
            if server is not None and server.poll() is not None:
                print(f"FAIL: server exited with {server.returncode}", file=sys.stderr)
                return 1
            if _get(a.url + "/healthz") == 500 or time.monotonic() - t0 > a.ready_timeout:
                print(f"FAIL: not ready (readyz {status})", file=sys.stderr)
                return 1
            time.sleep(2)
        print(f"ready after {time.monotonic() - t0:.0f} s")

        body = json.dumps({"items": [{"key": "smoke", "image": _crop(), "task": "formula"}]}).encode()
        req = urllib.request.Request(a.url + "/v1/formulas", data=body,  # noqa: S310
                                     headers={"Content-Type": "application/json"})
        t1 = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
            out = json.load(r)
        result = out["results"][0]
        print(f"{out['model']}: {result} ({time.monotonic() - t1:.1f} s)")
        if result.get("error") or not (result.get("text") or "").strip():
            print("FAIL: no text for the formula crop", file=sys.stderr)
            return 1
        return 0
    finally:
        if server is not None:
            server.terminate()
            server.wait(30)


if __name__ == "__main__":
    sys.exit(main())
