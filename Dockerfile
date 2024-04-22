ARG BASE=11.5.0-devel-ubuntu20.04
FROM nvidia/cuda:${BASE}

ARG GENN_VER
LABEL maintainer="J.C.Knight@sussex.ac.uk" \
    version=${GENN_VER} \
    org.opencontainers.image.documentation="https://genn-team.github.io/" \
    org.opencontainers.image.source="https://github.com/genn-team/genn" \
    org.opencontainers.image.title="GeNN Docker image"

# Update APT database and upgrade any outdated packages and install Python, pip and pkgconfig
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -yq --no-install-recommends python3-dev python3-pip swig gosu nano libffi-dev

# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn

# Set python3 to be the default version of python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip itself and install numpy and jupyter
RUN python -m pip install --upgrade pip && \
    python -m pip install numpy jupyter matplotlib psutil pybind11

# Copy GeNN into /opt
COPY . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# Install GeNN and PyGeNN
RUN python3 setup.py develop

# Start entrypoint
# **NOTE** in 'exec' mode shell arguments aren't expanded so can't use environment variables
ENTRYPOINT ["/opt/genn/bin/genn-docker-entrypoint.sh"]

# Default command will be to launch bash
CMD ["/bin/bash"]
