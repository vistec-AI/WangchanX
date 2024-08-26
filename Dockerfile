# FROM  nvidia/cuda:12.1.0-base-ubuntu20.04
FROM python:3.12.3-bookworm

# ENV TZ=Asia/Bangkok \
#     DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# ARG CUDA_HOME=/usr/local/cuda

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /project

COPY requirements.txt .

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    htop \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*
# download cuda
# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
# && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#   tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# install cuda
# RUN apt-get update && apt-get install -y nvidia-container-toolkit


RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# COPY /usr/local/cuda /project/cuda
#

# Download cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-4

ENV CUDA_HOME=/usr/local/cuda-12.4 
RUN pip3 install flash-attn --no-build-isolation
