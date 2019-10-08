FROM tensorflow/tensorflow:latest-gpu-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y caffe-cuda python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
