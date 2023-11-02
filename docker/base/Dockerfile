FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq
RUN apt-get install -y git vim
RUN apt-get install -y libgtk2.0-dev
RUN rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

COPY . /workspace