@echo off
echo Stopping and removing existing HR Assistant containers...

:: Stop and remove standalone containers
docker stop hr-assistant-container 2>NUL || echo No standalone container running
docker rm hr-assistant-container 2>NUL || echo No container to remove

:: Stop any docker-compose services
docker-compose down 2>NUL || echo No docker-compose services running

:: List any remaining containers that might conflict
echo Checking for any remaining containers using port 8501...
docker ps -a | findstr "8501"
docker ps -a | findstr "hr-assistant"

echo Cleanup completed. You can now run the Jenkins pipeline safely.
pause
