"""The service with a stub engine (moved from evilflowers-books-digitalizer
``tests/studio/batch/test_vlm_http.py``; the client half stayed there)."""

from __future__ import annotations

import base64
import io
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from evilflowers_vlm.app import create_app


class StubEngine:
    name = "stub"

    def __init__(self, gate: threading.Event | None = None, fail_load: bool = False) -> None:
        self.gate = gate
        self.fail_load = fail_load
        self.seen: list[tuple[tuple[int, int], str]] = []

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("no weights")

    def recognize(self, image: Image.Image, task: str) -> str:
        if self.gate is not None:
            self.gate.wait(5)
        self.seen.append((image.size, task))
        if image.size[0] == 99:
            raise RuntimeError("model blew up")
        return f"{task}:{image.size[0]}x{image.size[1]}"


def _png(w: int, h: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _metric(client: TestClient, line_prefix: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_prefix + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_service_recognizes_with_per_item_errors() -> None:
    engine = StubEngine()
    with TestClient(create_app(engine, load_in_background=False)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json()["status"] == "ready"
        resp = client.post("/v1/formulas", json={"items": [
            {"key": "a", "image": _png(200, 80)},
            {"key": "b", "image": _png(99, 99), "task": "ocr"},
            {"key": "c", "image": "not-base64!!"},
            {"key": "d", "image": base64.b64encode(b"not a png").decode()},
            {"key": "e", "image": _png(3, 140)}]})
    assert resp.status_code == 200
    by_key = {r["key"]: r for r in resp.json()["results"]}
    assert by_key["a"]["text"] == "formula:200x80"
    assert by_key["b"]["error"] == "model blew up"
    assert "base64" in by_key["c"]["error"] and "unreadable" in by_key["d"]["error"]
    assert by_key["e"]["text"] == "formula:64x140"      # normalized again server-side
    assert resp.json()["model"] == "stub"


def test_service_not_ready_until_loaded_and_unhealthy_on_load_failure() -> None:
    with TestClient(create_app(StubEngine(fail_load=True), load_in_background=False)) as client:
        assert client.get("/healthz").status_code == 500
        assert client.get("/readyz").status_code == 503
        resp = client.post("/v1/formulas", json={"items": [{"key": "a", "image": _png(9, 9)}]})
        assert resp.status_code == 503 and "Retry-After" in resp.headers


def test_service_rejects_bad_requests() -> None:
    with TestClient(create_app(StubEngine(), load_in_background=False)) as client:
        assert client.post("/v1/formulas", json={"items": []}).status_code == 422
        too_many = [{"key": str(i), "image": _png(9, 9)} for i in range(33)]
        assert client.post("/v1/formulas", json={"items": too_many}).status_code == 422
        bad_task = [{"key": "a", "image": _png(9, 9), "task": "table"}]
        assert client.post("/v1/formulas", json={"items": bad_task}).status_code == 422


def test_service_rejects_oversized_image_per_item() -> None:
    huge = base64.b64encode(b"\0" * (8 * 1024 * 1024 + 1)).decode()
    with TestClient(create_app(StubEngine(), load_in_background=False)) as client:
        resp = client.post("/v1/formulas", json={"items": [{"key": "a", "image": huge}]})
    assert resp.status_code == 200
    assert "larger than" in resp.json()["results"][0]["error"]


def test_service_queue_limit_returns_503() -> None:
    gate = threading.Event()
    engine = StubEngine(gate=gate)
    body = {"items": [{"key": "a", "image": _png(10, 10)}]}
    with TestClient(create_app(engine, max_queue=1, load_in_background=False)) as client:
        busy_before = _metric(client, 'vlm_queue_rejections_total{reason="busy"}')
        results: list[int] = []
        threads = [threading.Thread(target=lambda: results.append(
            client.post("/v1/formulas", json=body).status_code)) for _ in range(2)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 5
        while client.app.state.vlm.admitted < 2 and time.monotonic() < deadline:  # type: ignore[attr-defined]
            time.sleep(0.01)
        busy = client.post("/v1/formulas", json=body)
        gate.set()
        for t in threads:
            t.join(5)
        busy_after = _metric(client, 'vlm_queue_rejections_total{reason="busy"}')
    assert busy.status_code == 503 and int(busy.headers["Retry-After"]) >= 1
    assert sorted(results) == [200, 200]
    assert busy_after == busy_before + 1


def test_metrics_count_items_by_task_and_outcome() -> None:
    with TestClient(create_app(StubEngine(), load_in_background=False)) as client:
        ok = 'vlm_items_total{outcome="ok",task="formula"}'
        err = 'vlm_items_total{outcome="error",task="ocr"}'
        count = 'vlm_item_seconds_count{task="formula"}'
        before = (_metric(client, ok), _metric(client, err), _metric(client, count))
        client.post("/v1/formulas", json={"items": [
            {"key": "a", "image": _png(20, 20)},
            {"key": "b", "image": _png(99, 99), "task": "ocr"}]})
        after = (_metric(client, ok), _metric(client, err), _metric(client, count))
    assert after == (before[0] + 1, before[1] + 1, before[2] + 1)
