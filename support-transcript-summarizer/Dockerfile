FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.8
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY summarizer ./summarizer
COPY ui.py ./
COPY examples ./examples
COPY .streamlit ./.streamlit
RUN useradd --create-home app && mkdir /app/data && chown app:app /app/data
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8010
CMD ["uvicorn", "summarizer.api:app", "--host", "0.0.0.0", "--port", "8010"]
