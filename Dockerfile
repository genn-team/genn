ARG BASE=11.5.0-devel-ubuntu20.04
FROM nvidia/cuda:${BASE}

LABEL maintainer="J.C.Knight@sussex.ac.uk"
LABEL version="5.0.0"
LABEL org.opencontainers.image.documentation="https://genn-team.github.io/"
LABEL org.opencontainers.image.source="https://github.com/genn-team/genn"
LABEL org.opencontainers.image.title="GeNN Docker image"

# Update APT database and upgrade any outdated packages
RUN apt-get update && \
    apt-get upgrade -y

# Install Python, pip and swig
RUN apt-get install -yq --no-install-recommends python3-dev python3-pip gosu nano

# Set CUDA environment variable
ENV CUDA_PATH=/usr/local/cuda

ENV GENN_PATH=/opt/genn

# Upgrade pip itself
RUN pip install --upgrade pip

# Install numpy and jupyter
RUN pip install numpy jupyter matplotlib psutil pybind11

# Copy GeNN into /opt
COPY  . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# Install GeNN and PyGeNN
RUN python3 setup.py develop

# Default command will be to launch bash
CMD ["/bin/bash"]

# Start entrypoint
# **NOTE** in 'exec' mode shell arguments aren't expanded so can't use environment variables
ENTRYPOINT ["/opt/genn/bin/genn-docker-entrypoint.sh"]
