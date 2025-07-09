pipeline {
    agent any

    environment {
        IMAGE_NAME = 'hr-assistant'
        CONTAINER_NAME = 'hr-assistant-container'
        PORT = '8501'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build --no-cache -t ${IMAGE_NAME} ."
                }
            }
        }
        stage('Stop and Remove Old Container') {
            steps {
                script {
                    sh "docker stop ${CONTAINER_NAME} || true"
                    sh "docker rm ${CONTAINER_NAME} || true"
                }
            }
        }
        stage('Run Docker Container') {
            steps {
                script {
                    sh "docker run -d --name ${CONTAINER_NAME} -p ${PORT}:8501 ${IMAGE_NAME}"
                }
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}