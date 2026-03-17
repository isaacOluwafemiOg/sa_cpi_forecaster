# Use a slim Python image for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies inside the container
RUN pip install -r requirements.txt

# Copy the rest of the application code
# This includes style.css
COPY style.css .
COPY main.py .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose the port streamlit UI will run on
EXPOSE 8080

# Command to run the UI
CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
