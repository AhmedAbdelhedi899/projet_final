FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for temporary files and ensure file permissions
RUN mkdir -p /tmp/streamlit /app/data && \
    chmod -R 777 /tmp/streamlit /app/data

# Create a file for the API key if it doesn't exist
RUN touch /app/api.txt && chmod 666 /app/api.txt

# Validate the script
RUN python -m py_compile hr_assistant_enhanced.py

EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Use the HR Assistant script
CMD ["streamlit", "run", "hr_assistant_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]