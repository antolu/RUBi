FROM pytorch/pytorch

ARG GPU
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN if [ "$GPU" = "nvidia" ]; then \
    cd /tmp; \
    git clone https://github.com/NVIDIA/apex; \
    cd apex; \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./; \
    fi
