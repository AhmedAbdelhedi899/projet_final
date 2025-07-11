# Jenkins Pipeline Error Fix Documentation

## Problem

The Jenkins pipeline encountered the following error:

```
docker: Error response from daemon: failed to set up container networking: driver failed programming external connectivity on endpoint hr-assistant-container: Bind for 0.0.0.0:8501 failed: port is already allocated
```

This error occurs when Jenkins tries to start the HR Assistant container, but port 8501 is already in use by another container or service.

## Solutions Implemented

We've made several changes to fix this issue:

### 1. Updated Jenkinsfile

- Simplified the pipeline to focus on essential stages
- Added error handling for port conflicts
- Implemented a fallback to alternative ports when the primary port is unavailable
- Improved container naming and cleanup between runs

### 2. Updated docker-compose.yml

- Made container names consistent with those used in the Jenkinsfile
- Added environment variable support for port configuration
- Ensured consistent application file names between Dockerfile and docker-compose.yml
- Improved network configuration

### 3. Created Cleanup Scripts

- Added `cleanup-containers.sh` for Linux/Mac users
- Added `cleanup-containers.bat` for Windows users
- These scripts safely stop and remove any conflicting containers

### 4. Created PORT-CONFLICT-GUIDE.md

- Detailed instructions for users to resolve port conflicts
- Multiple approaches depending on the specific situation
- Instructions for accessing the services on alternative ports

## How to Use

### Before Running the Pipeline

1. Run the appropriate cleanup script for your OS:
   - Windows: `cleanup-containers.bat`
   - Linux/Mac: `chmod +x cleanup-containers.sh && ./cleanup-containers.sh`

2. Verify that no containers are using port 8501:
   ```
   docker ps -a | grep 8501
   ```

### When Running the Pipeline

The updated Jenkinsfile now:
1. Automatically stops and removes any existing containers
2. Attempts to start the container on the default port (8501)
3. If that fails, tries an alternative port (9501)
4. Provides clear output about which port to use to access the application

### After Pipeline Completion

Access the HR Assistant at one of:
- http://localhost:8501 (if using default port)
- http://localhost:9501 (if using alternative port)

## Further Improvements

- Consider implementing dynamic port assignment
- Add more robust health checks
- Implement a more sophisticated container naming strategy to avoid conflicts
- Set up Docker network isolation for improved security

## Contact

If you encounter any issues with this implementation, please refer to the PORT-CONFLICT-GUIDE.md file or contact the DevOps team.
