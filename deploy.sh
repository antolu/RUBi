#!/bin/bash
#
# Speedy script to install Docker and Nvidia-Docker
# and deploy the RUBi code inside a tensorflow 
# Docker container. 
#
# ........................................................

# Script has been curl'd, clone repo to current folder and deploy
if [[ `basename $PWD` != "RUBi" ]]; then
    git clone https://github.com/antolu/RUBi.git
    cd RUBi
    ./deploy.sh
    exit 0
fi

if [[ `lspci | grep Nvidia` != "" ]]; then
    gpu == "-gpu"
fi

if [[ `command -v docker` == "" ]]; then
    echo "Docker not installed. Installing Docker and Nvidia-Docker"
    sudo apt update
    sudo apt install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker

    if [[ ! -z $gpu && -z `command -v nvidia-docker` ]]; then 
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt install -y nvidia-container-toolkit
    fi
fi

if [[ `docker ps | grep tensorflow` != "" ]]; then
    echo "Stopping and removing existing container, press CTRL-C within 5 secs to cancel"
    for ((i=5; i>=1; i--)); do
    	echo $i
	sleep 1
    done

    docker stop tensorflow
    docker container rm tensorflow
fi

if [[ -z `docker images -q tensorflow/tensorflow:latest$gpu` ]]; then
    docker pull tensorflow/tensorflow:latest$gpu
fi

DOCKERARGS="-tid -p 8888:8888 --name tensorflow -v $PWD:/home/RUBi"

if [[ ! -z $gpu ]]; then
    DOCKERARGS+="--gpus all"
fi
docker run $DOCKERARGS tensorflow/tensorflow:latest$gpu

echo -e "\nThe Docker container is now online with the RUBi repo in /home/RUBi."
echo 'Execute commands inside the container as <docker exec -it -w /home/RUBi -u $(id -u):$(id -g) bash -c "python3 ...">'
