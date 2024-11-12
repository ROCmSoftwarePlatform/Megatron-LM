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

def clean_up_docker_containers() {
    sh """
    set +e
    docker stop ${env.CONTAINER_NAME} || true
    docker rm ${env.CONTAINER_NAME} || true
    """
}

def clean_up_docker_images() {
    sh """
    set +e
    docker rmi ${env.REPO_NAME}:${env.DOCKER_TAG} || true
    docker rmi ${env.REPO_NAME}:${env.imageSha} || true
    """
}

def clean_docker_build_cache() {
    sh 'docker system prune -f --volumes || true'
}

pipeline {
    agent { label 'build-only' }

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
                show_node_info()
                clean_up_docker_images()
                clean_docker_build_cache()
                script {
                    DOCKER_BUILD_ARGS = "--build-arg PYTORCH_ROCM_ARCH_OVERRIDE=${params.GPU_ARCH}"
                    sh "docker build -f Dockerfile_rocm.ci -t ${env.REPO_NAME}:${env.DOCKER_TAG} ${DOCKER_BUILD_ARGS} ."
                }
            }
        }

        stage('Tag Docker Image') {
            steps {
                script {
                    // Get the short image SHA (first 12 characters of the image ID)
                    env.imageSha = sh(script: "docker images --format '{{.ID}}' ${env.REPO_NAME}:${env.DOCKER_TAG} | head -c 12", returnStdout: true).trim()

                    if (!env.imageSha) {
                        error "Failed to retrieve the image SHA for ${env.REPO_NAME}:${env.DOCKER_TAG}"
                    }

                    echo "Image SHA: ${env.imageSha}"
                    echo "Docker Tag: ${env.DOCKER_TAG}"

                    // Tag the image with the short SHA
                    sh "docker tag ${env.REPO_NAME}:${env.DOCKER_TAG} ${env.REPO_NAME}:${env.imageSha}"
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "docker push ${env.REPO_NAME}:${env.imageSha}"  // Also push the image with short SHA
                    }
                }
            }
        }

        stage('Run Unit Tests') {
            agent { node { label "${params.TEST_NODE_LABEL}" } }
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
        }
    }

    post {
        always {
            // Archive test results
            archiveArtifacts artifacts: '**/test_report.csv', allowEmptyArchive: true

            script {
                def currentNodeLabels = env.NODE_LABELS ? env.NODE_LABELS.split() : []

                if (!currentNodeLabels.contains('build-only')) {
                    clean_up_docker_containers()
                }

                clean_up_docker_images()
            }
        }
    }
}
