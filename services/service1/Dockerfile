FROM python:3.11-slim

WORKDIR /app

COPY services/service1/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/service1/backend/ ./backend/
COPY services/service1/frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
