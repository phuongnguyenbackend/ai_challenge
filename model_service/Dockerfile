# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

COPY ./model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./model_service/NotoSans-Regular.ttf .
COPY ./model_service ./model_service

EXPOSE 8000

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8000"] 