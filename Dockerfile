FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shipment_api.py .
COPY delivery_pipeline.pkl .
EXPOSE 8000
CMD ["uvicorn", "shipment_api:app", "--host", "0.0.0.0", "--port", "8000"]
