FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including C++ build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ .
COPY embeddings/ ./embeddings/
COPY youtube.csv .
COPY utils.py .
COPY secrets.json .
COPY config.json .
COPY index.ann .
COPY channel_id_map.json .
COPY embeddings_generator.py .

EXPOSE 5000

CMD ["python", "service.py"]