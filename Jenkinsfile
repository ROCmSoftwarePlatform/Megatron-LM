import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}
DOCKER_IMAGE = megatron-lm
CONTAINER_NAME = megatron-lm-container
TEST_COMMAND = "bash ./run-tests.sh"
pipeline {

    agent {node {label "${params.TEST_NODE}"}}

    parameters {
        string(name: 'TEST_NODE', defaultValue: 'scm', description: 'Node or Label to launch Jenkins Job')
    }

    environment {
        DOCKER_WORKSPACE = "${env.WORKSPACE}/docker_workspace"
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    // Copy the necessary files into the Docker workspace
                    // sh "cp Dockerfile_amd ${DOCKER_WORKSPACE}/"
                    // dir(DOCKER_WORKSPACE) {
                        // Build Docker image
                    sh "docker build  -f Dockerfile_amd -t ${params.DOCKER_IMAGE} ."
                    }
                }
            }

        stage('Run Docker Container') {
            steps {
                script {
                    // Run the Docker container with the specified name
                    sh "docker run -d --name ${params.CONTAINER_NAME} ${params.DOCKER_IMAGE}"
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Execute test command in the running container
                    sh "docker exec ${params.CONTAINER_NAME} bash -c ${params.TEST_COMMAND}"
                }
            }
        }

         stage('Cleanup') {
            steps {
                script {
                    // Execute test command in the running container
                    sh "docker stop ${params.CONTAINER_NAME}"
                    sh "docker rm ${params.CONTAINER_NAME}"
                    sh "docker rmi ${params.DOCKER_IMAGE}"
                }
            }
        }

    }
}
