ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=22.04
# Define base image.
FROM mambaorg/micromamba:jammy-cuda-11.7.1
ARG CUDA_VERSION
ARG OS_VERSION
ENV CUDA_HOME="/usr/local/cuda"


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ARG CUDA_ARCHITECTURES=86

USER root
RUN apt update --fix-missing
RUN apt install -y cuda-toolkit-11-7 \
git \
cmake \
wget \
unzip \
ninja-build \
build-essential \
libeigen3-dev \
libglew-dev libgl1-mesa-dev \
python3-opencv \
libopenexr-dev \
libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev liblapacke-dev \
libxvidcore-dev libx264-dev \
libatlas-base-dev gfortran \
ffmpeg \
libopenh264-dev \
libopencv-dev=4.5.4+dfsg-9ubuntu4

# Verify CUDA version
RUN nvcc --version


# Copy files to the workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yaml /workspace/environment.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER ./submodules /workspace/submodules
WORKDIR /workspace



ENV TCNN_CUDA_ARCHITECTURES=86
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

RUN micromamba install -y -n base -f ./environment.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY --chown=$MAMBA_USER:$MAMBA_USER ./ /workspace/
USER $MAMBA_USER
