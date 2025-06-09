# Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# Set environment variables to prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for Kaleido
RUN apt-get update && apt-get install -y \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo contents
COPY . .

# Make sure the pipeline script is executable
RUN chmod +x pipeline/run_pipeline.bash

# Run the full pipeline when the container starts
CMD ["bash", "pipeline/run_pipeline.bash"]
