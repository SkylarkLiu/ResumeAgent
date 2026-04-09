FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG DEBIAN_MIRROR=mirrors.aliyun.com
ARG DEBIAN_MIRROR_SCHEME=http
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_FALLBACK_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=180
ARG PIP_RETRIES=10

COPY requirements.txt .
RUN sh -c '\
    pip install \
        --no-cache-dir \
        --disable-pip-version-check \
        --prefer-binary \
        --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        -i "${PIP_INDEX_URL}" \
        -r requirements.txt \
    || pip install \
        --no-cache-dir \
        --disable-pip-version-check \
        --prefer-binary \
        --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        -i "${PIP_FALLBACK_INDEX_URL}" \
        -r requirements.txt'

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
