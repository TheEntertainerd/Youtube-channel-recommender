FROM python:3.9-slim

WORKDIR /app

COPY dash/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dash/ .

EXPOSE 8050

CMD ["python", "app.py"]