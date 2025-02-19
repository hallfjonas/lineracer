# Use Ubuntu 24.04 LTS as the base image
FROM ubuntu:20.04
LABEL maintainer="emilcode"

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install dev tools
RUN apt-get update && apt-get install -y --no-install-recommends\
    python3 \
    python3-pip \
    git \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install python packages listed in the requirements.txt
# RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
#     pip3 install --no-cache-dir -r /tmp/requirements.txt

# # Create a non-root user and switch to it
# RUN groupadd -g 1000 devgroup && \
#     useradd -m -u 1000 -g devgroup devuser \
#     chown -R devuser:devuser /workspace/lineracer

# # Switch to the custom user
# USER devuser

# # Set the workdir
# WORKDIR /workspace/

# # Add the local user's .local bin directory to the PATH
# ENV PATH="/home/devuser/.local/bin:${PATH}"

# Install flit to create and install python package
RUN pip install flit

# Set environment variable to allow flit install to install as root user
ENV FLIT_ROOT_INSTALL=1