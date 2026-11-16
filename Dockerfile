# evilflowers-vlm-service: PaddleOCR-VL formula service, CPU-only, weights baked in.
#
#   docker build -t evilflowers-vlm-service .
#   docker run --rm -p 8000:8000 --read-only --tmpfs /tmp -e VLM_THREADS=4 evilflowers-vlm-service
#
# Weights are BAKED IN at a pinned Hugging Face revision rather than fetched into a PVC by an
# init container: the image is immutable and self-describing (the tag pins code + weights
# together), the pod needs no network, no PVC and works with readOnlyRootFilesystem; the cost
# is ~1.9 GB more per pull, once per node. Only the safetensors weights, JSON configs,
# tokenizer and chat template are fetched: no pickled checkpoints and no remote code (*.py).
# The model class is native in transformers >= 5, and nothing passes trust_remote_code.
#
# Runtime contract: any non-root uid:gid works (k8s runs 65534); nothing is written outside
# /tmp (HOME), so readOnlyRootFilesystem with an emptyDir on /tmp is enough.

ARG PYTHON_IMAGE=python:3.12-slim-trixie

# --- weights: independent of the code, so a code change never re-downloads them ----------
FROM ${PYTHON_IMAGE} AS weights
ARG VLM_REPO=PaddlePaddle/PaddleOCR-VL-1.6
ARG VLM_REVISION=132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install huggingface_hub==2.4.0
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${VLM_REPO}', revision='${VLM_REVISION}', local_dir='/models/PaddleOCR-VL-1.6', \
    allow_patterns=['*.safetensors', '*.json', 'tokenizer.model', 'chat_template.jinja', 'LICENSE'])" \
 && rm -rf /models/PaddleOCR-VL-1.6/.cache \
 && test -f /models/PaddleOCR-VL-1.6/model.safetensors \
 && ! find /models -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.py' | grep . \
 && chmod -R a+rX /models

# --- venv: CPU torch + the package, from uv.lock ---------------------------------------
FROM ${PYTHON_IMAGE} AS venv
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra model --extra otel
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra model --extra otel \
 && /opt/venv/bin/python -c "import torch, transformers; assert not torch.cuda.is_available(); print(torch.__version__, transformers.__version__)" \
 && chmod -R a+rX /opt/venv

# --- runtime -------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
COPY --from=venv /opt/venv /opt/venv
COPY --from=weights /models /models
RUN useradd --uid 10001 --gid 0 --home-dir /tmp --no-create-home --shell /usr/sbin/nologin vlm
ENV PATH=/opt/venv/bin:$PATH \
    VLM_MODEL_DIR=/models/PaddleOCR-VL-1.6 \
    VLM_DEVICE=cpu \
    LOG_FORMAT=json \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /tmp
USER 10001
EXPOSE 8000
# k8s ignores HEALTHCHECK and uses its own probes; python, not curl (the image has none).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"]
CMD ["python", "-m", "evilflowers_vlm"]
