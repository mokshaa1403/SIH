FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY satquery ./satquery
COPY config ./config
EXPOSE 8000
CMD ["python", "-m", "satquery.server", "--host", "0.0.0.0", "--port", "8000"]

