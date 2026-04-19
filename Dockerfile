# Owner: Syed Ibrahim Saleem
# MemeRAG — Dockerfile

FROM python:3.10-slim

WORKDIR /app

# libgomp1 required by sentence-transformers/PyTorch
# curl required by HEALTHCHECK
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --trusted-host flags prevent hash mismatch errors on PyPI downloads
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# create runtime directories so bind mounts never throw FileNotFoundError
RUN mkdir -p /app/demo_images /app/data /app/chroma_db

COPY . .

EXPOSE 8501

# healthcheck so Docker knows when Streamlit is actually ready
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
