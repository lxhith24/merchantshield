FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/uploads && chmod +x deploy/entrypoint.sh

ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces (Docker SDK) routes this one port publicly; the
# FastAPI process stays internal on 127.0.0.1:8000.
EXPOSE 7860

CMD ["/app/deploy/entrypoint.sh"]
