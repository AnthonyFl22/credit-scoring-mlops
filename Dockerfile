FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer so it's cached on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs at runtime
COPY src/ ./src/
COPY api/ ./api/
COPY models/credit_scoring/v1/ ./models/credit_scoring/v1/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
