
# Use Python 3.11 as the base environment.
# Change this version if your local Python version differs.
FROM python:3.11-slim

# Set application working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy saved ML model
COPY models/ ./models/

# API listens on port 8000
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]