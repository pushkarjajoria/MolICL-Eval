# for LSV A100s server
FROM nvcr.io/nvidia/pytorch:22.02-py3

# for LSV V100 server
# FROM nvcr.io/nvidia/pytorch:21.07-py3

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
# RUN SHA=ToUcHMe which python3
# RUN python3 -m pip install --upgrade pip


# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies (this is not necessary when using an *external* mini conda environment)



# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME


# Setup VSCode stuff (comment when not using vscode)
# RUN mkdir /home/$USER_NAME/.vscode-server
# RUN mkdir /home/$USER_NAME/.vscode-server-insiders

# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local

# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]