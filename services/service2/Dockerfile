FROM python:3.11-slim

WORKDIR /app

COPY services/service2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/service2/ .

# Copy only the script file needed from linesight — no RL training code in the container
COPY linesight/scripts/extract_driver_profile.py ./scripts/extract_driver_profile.py

ENV LINESIGHT_SCRIPTS_PATH=/app/scripts

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
