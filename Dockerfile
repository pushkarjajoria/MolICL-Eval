# Base image with PyTorch 2.5.0 and CUDA 12.1
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN python3 -m pip install --upgrade pip

# Set locale
ENV LANG C.UTF-8

# Install dependencies
RUN python3 -m pip install numpy autopep8

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME && \
    useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME && \
    mkdir /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]
