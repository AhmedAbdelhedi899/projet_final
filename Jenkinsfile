pipeline {
    agent any
    
    environment {
        // Application settings
        APP_NAME = 'hr-assistant'
        APP_PORT = '8501'
        APP_CONTAINER_NAME = 'hr-assistant-container'
        
        // Docker settings
        DOCKER_COMPOSE_FILE = 'docker-compose.yml'
        
        // Groq API key - preferably stored as a Jenkins credential
        GROQ_API_KEY = credentials('groq-api-key') // Create this credential in Jenkins
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Checked out branch: ${env.BRANCH_NAME ?: 'main'}"
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building HR Assistant Docker image..."
                    sh "docker build --no-cache -t ${APP_NAME} ."
                }
            }
        }
        
        stage('Stop and Remove Old Container') {
            steps {
                script {
                    echo "Stopping and removing existing containers if they exist..."
                    
                    // Stop and remove the standalone container if it exists
                    sh "docker stop ${APP_CONTAINER_NAME} || true"
                    sh "docker rm ${APP_CONTAINER_NAME} || true"
                    
                    // Stop the docker-compose services if they exist
                    sh "docker-compose -f ${DOCKER_COMPOSE_FILE} down || true"
                }
            }
        }
        
        stage('Run Docker Container') {
            steps {
                script {
                    echo "Starting HR Assistant container..."
                    
                    // Try to start the container with the default port
                    def containerStart = sh(script: "docker run -d --name ${APP_CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${APP_NAME}", returnStatus: true)
                    
                    // If the default port is in use, try an alternative port
                    if (containerStart != 0) {
                        echo "Default port ${APP_PORT} is in use, trying alternative port 9501..."
                        sh "docker run -d --name ${APP_CONTAINER_NAME} -p 9501:${APP_PORT} ${APP_NAME}"
                        echo "HR Assistant is now available at http://localhost:9501"
                    } else {
                        echo "HR Assistant is now available at http://localhost:${APP_PORT}"
                    }
                }
            }
        }
    }
    
    post {
        success {
            echo "HR Assistant deployment completed successfully!"
        }
        
        failure {
            echo "HR Assistant deployment failed. See logs for details."
            sh "docker logs ${APP_CONTAINER_NAME} || true"
        }
        
        always {
            cleanWs(cleanWhenNotBuilt: false,
                    deleteDirs: true,
                    disableDeferredWipeout: true,
                    patterns: [[pattern: '.git/**', type: 'EXCLUDE']])
            
            echo "Pipeline completed with status: ${currentBuild.result}"
        }
    }
}