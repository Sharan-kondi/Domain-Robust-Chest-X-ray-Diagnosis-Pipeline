FROM python:3.11-slim

# Set environment variables for clean logs and Python performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV/PIL image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first to keep image sizes small (saving ~1.2GB)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and agent pipeline code directories
COPY src/ ./src/
COPY serving/ ./serving/
COPY radagent/ ./radagent/
COPY configs/ ./configs/

# Expose port and run server
EXPOSE 8000

CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port $PORT"]
