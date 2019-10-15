FROM pytorch/pytorch

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
