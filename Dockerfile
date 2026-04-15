FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -e "."
COPY . .
RUN uv pip install --system -e "."

WORKDIR /work

ENTRYPOINT ["rotten"]
