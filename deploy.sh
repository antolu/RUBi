#!/bin/bash
#
# Speedy script to install Docker and Nvidia-Docker
# and deploy the RUBi code inside a tensorflow 
# Docker container. 
#
# ........................................................

set -e

#............................................................
#
# Installs the required packages (i.e. Docker and
#    nvidia-container-toolkit)
#
#............................................................
installPackages() {
    OLD_DIR=$PWD
    cd /tmp
    
    source /etc/os-release
    if [[ -z `command -v docker` ]]; then
	echo "Installing Docker"
	case $ID in
	    arch)
		sudo pacman -Sy --needed --noconfirm docker
		;;
	    ubuntu|debian)
		sudo apt update
		sudo apt install -y docker
		;;
	esac
	sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    fi

    if [[ $GPU == "nvidia" ]]; then
	echo "Installing Nvidia-Docker and CUDA toolkit"
	case $ID in
	    arch)
		sudo pacman -Sy --needed --noconfirm cuda
		
		git clone https://aur.archlinux.org/libnvidia-container.git
		cd libnvidia-container
		makepkg -sci --needed --noconfirm
		cd ..

		git clone https://aur.archlinux.org/nvidia-container-toolkit.git
		cd nvidia-container-toolkit
		makepkg -sci --needed --noconfirm
		cd ..

		rm -rf libnvidia-container nvidia-container-toolkit
		;;
	    ubuntu|debian)
		sudo apt install -y nvidia-cuda-toolkit nvidia-container-toolkit
		;;
	esac
	sudo systemctl restart docker
    fi

    cd $OLD_DIR
}

#............................................................
#
# Parses command line arguments 
#
#............................................................
parseArguments() {
    # defaults
    GPU="none"
    REPO="antolu"
    BRANCH="master"

    POSITIONAL=()
    while [[ $# -gt 0 ]]
    do
	key="$1"

	case $key in
	    --runtime)
		GPU="$2"
		shift # past argument
		shift # past value
		;;
	    -r|--repo)
		REPO="$2"
		shift # past argument
		shift # past value
		;;
	    -b|--branch)
		BRANCH="$2"
		shift # past argument
		shift # past value
		;;
	    *)    # unknown option
		POSITIONAL+=("$1") # save it in an array for later
		shift # past argument
		;;
	esac
    done

    if [[ $GPU == "none" && ! -z `lspci | grep -i nvidia` ]]; then
	GPU="nvidia"
    fi

}

#............................................................
#
# Builds the Docker images containing TF and Caffe (and any
#    other packages specified in the Dockerfiles.
#
#............................................................
buildTFImage() {
    if [[ $GPU == "nvidia" && -z `docker images -q tf-gpu:latest` ]]; then
	docker build --file ./Dockerfile/tf-gpu.Dockerfile -t tf-gpu:latest .
    elif [[ -z `docker images -q tf-cpu:latest` ]]; then
	docker build --file ./Dockerfile/tf-cpu.Dockerfile -t tf-cpu:latest .
    fi
}

#............................................................
#
# Removes the existing TF container (by name) to make room
#   for a new one.
#
#............................................................
removeTFContainer() {
    if [[ `docker ps | grep tf-rubi` != "" ]]; then
	echo "Stopping and removing existing container, press CTRL-C within 5 secs to cancel"
	for ((i=5; i>=1; i--)); do
    	    echo $i
	    sleep 1
	done
	
	docker stop tf-rubi
	docker container rm tf-rubi
    fi
}

#............................................................
#
# Runs a new Tensorflow container, gpu or cpu depending on
#   environment variable GPU
#
#............................................................
runTFContainer() {
    DOCKERARGS="-tid -p 8888:8888 --name tf-rubi -v $PWD:/home/RUBi"
    
    if [[ $GPU == "nvidia" ]]; then
	DOCKERARGS+=" --gpus all"
	docker run $DOCKERARGS tf-gpu:latest
    else
	docker run $DOCKERARGS tf-cpu:latest
    fi
}

parseArguments $@

# Script has been curl'd, clone repo to current folder and deploy
if [[ `basename $PWD` != "RUBi" ]]; then
    git clone https://github.com/$REPO/RUBi.git
    cd RUBi
    git checkout $BRANCH
    
    installPackages
    buildTFImage
    
    ./deploy.sh
    exit 0
fi

removeTFContainer

runTFContainer

echo -e "\nThe Docker container is now online with the RUBi repo in /home/RUBi."
echo 'Execute commands inside the container as <docker exec -it -w /home/RUBi -u $(id -u):$(id -g) tf-rubi bash -c "python3 ...">'
