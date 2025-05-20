# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /model_service

COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 100 --retries 3 -r requirements.txt

COPY NotoSans-Regular.ttf ./
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 