FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY mlops_platform/ mlops_platform/
COPY configs/ configs/
COPY templates/batch_inference/ templates/batch_inference/

CMD ["python", "-m", "templates.batch_inference.scorer"]
