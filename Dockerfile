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

EXPOSE 8000

# 同一镜像两种角色（架构 §4.2）：
#   API    默认 CMD
#   Worker docker run ... arq app.worker.Settings   （M1 引入）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
