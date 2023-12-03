ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV DEBIAN_FRONTEND=noninteractive

# To fix GPG key error when running apt-get update (https://github.com/NVIDIA/nvidia-docker/issues/1631)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*
RUN apt-get install -y gcc
RUN pip --no-cache-dir install Cython

WORKDIR /workspace
COPY . /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt
RUN pip install shapely
RUN pip install opencv-python

RUN pip install tensorboard && \
    pip install timm==0.3.2 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html install torchtext==0.9.1
