# Build from the repo root: docker build -f api/Dockerfile -t legal-retrieval-api .
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image small; the rest comes from requirements.txt.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

# Artifacts are mounted at run time: -v ./artifacts:/app/artifacts
ENV ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
