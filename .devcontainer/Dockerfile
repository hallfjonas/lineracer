# Use Ubuntu 24.04 LTS as the base image
FROM ubuntu:20.04
LABEL maintainer="emilcode"

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set FLIT_ROOT_INSTALL to allow root install
ENV FLIT_ROOT_INSTALL=1

# Update the package list and install dev tools
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    git \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install python packages listed in the requirements.txt
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip3 install --no-cache-dir -r /tmp/requirements.txt