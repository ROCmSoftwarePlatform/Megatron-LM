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

DOCKER_IMAGE = "megatron-lm"
CONTAINER_NAME = "megatron-lm-container"
DOCKER_ARGS = "--workdir /workspace/Megatron-LM --entrypoint /workspace/Megatron-LM/run_tests.sh"
DOCKER_RUN_CMD= "docker run --rm -t -u 1069:1071 --network host -u root --group-add video --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host --env DLM_MODEL_NAME='pyt_huggingface_gpt_neo' --env JENKINS_BUILD_NUMBER='0' --device=/dev/kfd --device=/dev/dri/renderD128  --device=/dev/dri/renderD128  --device=/dev/dri/renderD129  --device=/dev/dri/renderD130  --device=/dev/dri/renderD131"
pipeline {

    parameters {
        string(name: 'TEST_NODE_LABEL', defaultValue: 'MI250', description: 'Node or Label to launch Jenkins Job')
    }
    
    agent {node {label "${params.TEST_NODE_LABEL}"}}

    stages {
        stage('Build Docker Image') {
            steps {
                show_node_info()
                script {
                    sh "docker build  -f Dockerfile_amd -t ${DOCKER_IMAGE} ."
                    }
                }
            }

        stage('Run Docker Container') {
            steps {
                script {
                    sh "${DOCKER_RUN_CMD} ${DOCKER_ARGS} --name ${CONTAINER_NAME} ${DOCKER_IMAGE} "
                }
            }
        }
    }

    post { 
        always { 
            //Cleanup
            script {
                sh "docker rmi ${DOCKER_IMAGE}"
            }
        }
    }
}
