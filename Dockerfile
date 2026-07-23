FROM node:20-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TICKET_QA_OUTPUT_DIR=/app/outputs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends fontconfig fonts-dejavu-core libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY workload_analysis ./workload_analysis
COPY sample_data ./sample_data
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/outputs

EXPOSE 8000
CMD ["python", "-m", "workload_analysis.server_with_upload"]
