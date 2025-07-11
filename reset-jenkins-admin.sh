#!/bin/bash
# Script to reset Jenkins admin password

# Stop the containers
docker-compose down

# Create a directory for the script
mkdir -p jenkins-init

# Create the initialization script
cat > jenkins-init/init.groovy.d/basic-security.groovy << 'EOF'
#!groovy
import jenkins.model.*
import hudson.security.*

def instance = Jenkins.getInstance()
def hudsonRealm = new HudsonPrivateSecurityRealm(false)
def adminUsername = "admin"
def adminPassword = "admin"
hudsonRealm.createAccount(adminUsername, adminPassword)
instance.setSecurityRealm(hudsonRealm)

def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(false)
instance.setAuthorizationStrategy(strategy)
instance.save()

println "Admin user created: " + adminUsername
EOF

# Update docker-compose.yml to mount the init scripts
# (This is temporary and will require manual editing)
echo "Please update your docker-compose.yml to include this volume mount for Jenkins:"
echo "  - ./jenkins-init/init.groovy.d:/var/jenkins_home/init.groovy.d"
echo "Then run 'docker-compose up -d' to restart with the admin user"
