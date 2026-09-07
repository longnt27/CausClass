FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache \
    CAUSCLASS_OUTPUT_DIR=/results WANDB_MODE=disabled
WORKDIR /app
RUN pip install --no-cache-dir uv==0.8.22
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY core ./core
COPY utils ./utils
COPY data ./data
RUN uv sync --locked --no-dev && \
    groupadd --gid 10001 research && \
    useradd --uid 10001 --gid research --create-home research && \
    mkdir -p /results && chown -R research:research /results /app /tmp/uv-cache
USER research
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "core.smoke"]
CMD ["--output-dir", "/results/smoke"]
