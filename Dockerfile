FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG DEBIAN_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org|https://${DEBIAN_MIRROR}|g; s|http://security.debian.org|https://${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org|https://${DEBIAN_MIRROR}|g; s|http://security.debian.org|https://${DEBIAN_MIRROR}|g" /etc/apt/sources.list; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
