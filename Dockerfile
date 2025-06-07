FROM python:3.13-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[terminal-bench]

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

