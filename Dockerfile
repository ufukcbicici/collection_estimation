# Collections forecasting (the hurdle model) on a compute machine, reached over SSH.
# Same layout as Dockerfile_old. The code and the data are NOT in the image: the project
# folder on the machine is mounted as the workspace, and PyCharm deploys the code into it.
#
# Before building, put the public key(s) allowed to log in next to this file, as
# `authorized_keys` (gitignored). Then:
#
#   docker build -t collection_estimation .
#   docker run -d --name collections -p 2222:22 \
#       -v /path/on/machine/collection_estimation:/workspace/collection_estimation \
#       collection_estimation
#   ssh -p 2222 appuser@<machine>
#
# PyCharm remote interpreter (SSH): /opt/conda/envs/collection_estimation/bin/python

# Base image: plain Ubuntu, not CUDA. The models are scikit-learn and LightGBM, on the CPU,
# so the CUDA image (Dockerfile_old's nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04) would
# only add several GB. Swap it back here if a GPU model (e.g. TiRex) is ever added.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ------------------------------------------------------------------------------
# 1. System dependencies & ssh
# ------------------------------------------------------------------------------
# libgomp1: the OpenMP runtime LightGBM's Linux wheel needs (scikit-learn bundles its own).
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    bzip2 \
    openssh-server \
    sudo \
    tmux \
    htop \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# sshd runtime dirs + host keys
RUN mkdir -p /run/sshd /var/run/sshd && \
    ssh-keygen -A

RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config

# ------------------------------------------------------------------------------
# 2. Install Miniconda
# ------------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH=$CONDA_DIR/bin:$PATH

# ------------------------------------------------------------------------------
# 3. Create non-root user for SSH
# ------------------------------------------------------------------------------
# -s /bin/bash: a --system user gets /bin/sh otherwise, which never reads ~/.bashrc.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} -s /bin/bash appuser -g sudo

RUN mkdir -p /home/appuser/.ssh
COPY authorized_keys /home/appuser/.ssh/authorized_keys

RUN chown -R appuser:sudo /home/appuser/.ssh && \
    chmod 700 /home/appuser/.ssh && \
    chmod 600 /home/appuser/.ssh/authorized_keys

EXPOSE 22

# Use bash -lc so "conda" is available in RUN steps
SHELL ["bash", "-lc"]

# ------------------------------------------------------------------------------
# 4. Conda environment
# ------------------------------------------------------------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n collection_estimation python=3.11 -y && \
    conda clean -afy

# The same pinned versions as the development machine (numpy, pandas, scikit-learn, scipy,
# matplotlib, openpyxl, joblib, pytest, lightgbm).
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n collection_estimation python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

ENV CONDA_DEFAULT_ENV=collection_estimation
ENV PATH=/opt/conda/envs/collection_estimation/bin:$PATH

# ENV is not seen by an SSH session (sshd starts it with a clean environment), so a login
# shell gets the same PATH and locale from here.
RUN printf '%s\n' \
    'export PATH=/opt/conda/envs/collection_estimation/bin:/opt/conda/bin:$PATH' \
    'export CONDA_DEFAULT_ENV=collection_estimation' \
    'export LANG=C.UTF-8 LC_ALL=C.UTF-8' \
    > /etc/profile.d/collection_estimation.sh

# ------------------------------------------------------------------------------
# 5. Workspace
# ------------------------------------------------------------------------------
# Owned by appuser, so PyCharm can deploy into it when no folder is mounted over it.
RUN mkdir -p /workspace/collection_estimation && \
    chown appuser:sudo /workspace/collection_estimation
WORKDIR /workspace/collection_estimation

# ------------------------------------------------------------------------------
# 6. Start SSH properly
# ------------------------------------------------------------------------------
CMD ["/usr/sbin/sshd", "-D"]
