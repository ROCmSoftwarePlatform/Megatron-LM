import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def clean_up_docker_images() {
    echo "Running clean_up_docker_images..."

    // Check if the images exist before attempting to remove them
    def imageExists = sh(script: "docker images -q ${env.REPO_NAME}:${env.DOCKER_TAG}", returnStdout: true).trim()
    def imageShaExists = sh(script: "docker images -q ${env.REPO_NAME}:${env.imageSha}", returnStdout: true).trim()

    if (imageExists) {
        echo "Removing Docker image: ${env.REPO_NAME}:${env.DOCKER_TAG}"
        sh "docker rmi ${env.REPO_NAME}:${env.DOCKER_TAG}"
    }

    if (imageShaExists) {
        echo "Removing Docker image: ${env.REPO_NAME}:${env.imageSha}"
        sh "docker rmi ${env.REPO_NAME}:${env.imageSha}"
    }
}

def clean_docker_build_cache() {
    sh 'docker system prune -f --volumes || true'
}

pipeline {
    agent {
        label 'build-only'
    }

    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'MI300X_BANFF', description: 'Node or Label to launch Jenkins Job')
        string(name: 'GPU_ARCH', defaultValue: 'gfx942', description: 'GPU Architecture')
    }

    environment {
        REPO_NAME = 'rocm/megatron-lm'
        DOCKER_TAG = 'latest'
        CONTAINER_NAME = "megatron-lm-container"
        DOCKER_RUN_ARGS = "-v \$(pwd):/workspace/Megatron-LM/output --workdir /workspace/Megatron-LM \
        --entrypoint /workspace/Megatron-LM/run_unit_tests.sh"
        DOCKER_RUN_CMD = "docker run --rm -t --network host -u root --group-add video --cap-add=SYS_PTRACE \
        --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
        --ipc=host --device=/dev/kfd --device=/dev/dri"
    }

    stages {
        stage('Build Docker Image') {
            steps {
                clean_docker_build_cache()
                script {
                    DOCKER_BUILD_ARGS = "--build-arg PYTORCH_ROCM_ARCH_OVERRIDE=${params.GPU_ARCH}"

                    // Build Docker image
                    sh "docker build -f Dockerfile_rocm.ci -t ${env.REPO_NAME}:${env.DOCKER_TAG} ${DOCKER_BUILD_ARGS} ."

                    // Get the short image SHA (first 12 characters of the image ID)
                    env.imageSha = sh(script: "docker images --format '{{.ID}}' ${env.REPO_NAME}:${env.DOCKER_TAG} | head -c 12", returnStdout: true).trim()
                    if (!env.imageSha) {
                        error "Failed to retrieve the image SHA for ${env.REPO_NAME}:${env.DOCKER_TAG}"
                    }

                    // Tag the image with the short SHA and push to repo
                    sh "docker tag ${env.REPO_NAME}:${env.DOCKER_TAG} ${env.REPO_NAME}:${env.imageSha}"
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker push ${env.REPO_NAME}:${env.imageSha}"  
                    }
                }
            }
            post {
                always {
                    clean_up_docker_images()
                }
            }
        }

        stage('Run Unit Tests') {
            agent {
                node {
                    label "${params.TEST_NODE_LABEL}"
                }
            }

            steps {
                script {
                    // Pull the Docker image from the repository on the test node
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker pull ${env.REPO_NAME}:${env.imageSha}"
                    }

                    wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'xterm']) {
                        sh "${DOCKER_RUN_CMD} ${DOCKER_RUN_ARGS} --name ${env.CONTAINER_NAME} ${env.REPO_NAME}:${env.imageSha}"
                    }
                }
            }
            post {
                always {
                // Archive test results
                script {
                    archiveArtifacts artifacts: 'test_report.csv', allowEmptyArchive: true
                    clean_up_docker_images()
                    }
                }
            }
        }
    }
}
