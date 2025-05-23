# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    fonts-noto-cjk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY ./model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./model_service/NotoSans-Regular.ttf .
COPY ./model_service .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 