# Use an official, lightweight Python base image.
# Python 3.11 satisfies the requirements of Keras 3.13.1 and TensorFlow 2.20.0.
FROM python:3.11-slim

# Set environment variables to optimize Python performance inside the container:
# - PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED=1: Prevents Python from buffering stdout/stderr (enables real-time logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container's virtual filesystem
WORKDIR /app

# Install basic system build dependencies. We clean up apt cache afterwards to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt first to leverage Docker's caching mechanism.
# If requirements.txt doesn't change, Docker will skip installing packages next time.
COPY requirements.txt .

# Upgrade pip and install Python packages defined in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the project files from the local directory (modular_pipeline) to the container (/app)
COPY . .

# Expose the default port that Streamlit uses (8501) so the network traffic can reach it
EXPOSE 8501

# Define the command to run your web application when the container starts.
# We configure it to run on port 8501 and bind to all network interfaces (0.0.0.0).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
