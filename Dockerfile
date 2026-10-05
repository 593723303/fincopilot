FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
# eval 与 frontend 也要进镜像：前者让评估能在容器里跑，
# 后者是 chainlit 演示界面，少了它镜像就只能当后端用
COPY eval ./eval
COPY frontend ./frontend
COPY chainlit.md ./

EXPOSE 8000 8001

# 健康探针指向 /healthz（只看进程存活）而不是 /readyz（要三个存储都通）。
# 用 readyz 做容器健康检查，会在存储短暂抖动时把本来能服务的实例判死。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# 同一镜像两种角色（架构 §4.2）：
#   API    默认 CMD
#   Worker docker run ... arq app.worker.Settings   （M1 引入）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
