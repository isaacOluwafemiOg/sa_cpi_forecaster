# Use a slim Python image for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY fastapi_requirements.txt .

# Install dependencies inside the container
RUN pip install -r fastapi_requirements.txt

# Copy the rest of the application code
# This includes src/, data/, models/
COPY data ./data
COPY models ./models
COPY src/main.py ./src/main.py

# Set PYTHONPATH to ensure imports like 'from src.forecaster.features' work correctly
ENV PYTHONPATH=/app

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the API
# We point to src.main:app because main.py is inside the src/ folder
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
