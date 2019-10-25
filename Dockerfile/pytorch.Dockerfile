FROM pytorch/pytorch

ARG GPU
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y python3-pip python python-scipy libsm6 libxext6 libxrender-dev libglib2.0-0 

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN if [ "$GPU" = "nvidia" ]; then \
    cd /tmp; \
    git clone https://github.com/NVIDIA/apex; \
    cd apex; \
    pip install -v --no-cache-dir --global-option="--cpp_ext" ./; \
    fi
