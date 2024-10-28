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

show_node_info()

pipeline {

    agent {node {label "${params.TEST_NODE}"}}

    parameters {
        string(name: 'TEST_NODE', defaultValue: 'scm', description: 'Node or Label to launch Jenkins Job')
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build  -f Dockerfile_amd -t ${params.DOCKER_IMAGE} ."
                    }
                }
            }

        stage('Run Docker Container') {
            steps {
                script {
                    sh "docker run -d --name ${params.CONTAINER_NAME} ${params.DOCKER_IMAGE}"
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    sh "docker exec ${params.CONTAINER_NAME} bash -c ${params.TEST_COMMAND}"
                }
            }
        }

         stage('Cleanup') {
            steps {
                script {
                    sh "docker stop ${params.CONTAINER_NAME}"
                    sh "docker rm ${params.CONTAINER_NAME}"
                    sh "docker rmi ${params.DOCKER_IMAGE}"
                }
            }
        }

    }
}
