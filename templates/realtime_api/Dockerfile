FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]"

COPY src/ src/
COPY mlops_platform/ mlops_platform/
COPY templates/realtime_api/ templates/realtime_api/

EXPOSE 8000

CMD ["uvicorn", "templates.realtime_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
