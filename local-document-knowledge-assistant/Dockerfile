FROM python:3.11-slim
WORKDIR /app
COPY requirements-core.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-core.lock
COPY knowledge_lab ./knowledge_lab
COPY evaluation ./evaluation
COPY examples ./examples
COPY ui.py ./
COPY scripts ./scripts
EXPOSE 8020
CMD ["python", "-m", "uvicorn", "knowledge_lab.api:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
