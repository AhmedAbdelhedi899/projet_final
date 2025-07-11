#!/bin/bash

echo "Stopping and removing existing HR Assistant containers..."

# Stop and remove standalone containers
docker stop hr-assistant-container || true
docker rm hr-assistant-container || true

# Stop any docker-compose services
docker-compose down || true

# List any remaining containers that might conflict
echo "Checking for any remaining containers using port 8501..."
docker ps -a | grep -E "8501|hr-assistant"

echo "Cleanup completed. You can now run the Jenkins pipeline safely."
