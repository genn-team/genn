ARG BASE=12.2.0-devel-ubuntu22.04
FROM nvidia/cuda:${BASE} AS build

LABEL maintainer="J.C.Knight@sussex.ac.uk" \
    version="4.8.1" \
    org.opencontainers.image.documentation="https://genn-team.github.io/" \
    org.opencontainers.image.source="https://github.com/genn-team/genn" \
    org.opencontainers.image.title="GeNN Docker image"

# Update APT database and upgrade any outdated packages and install Python, pip and swig
RUN apt-get update && \
    apt-get upgrade -y &&\
    apt-get install -yq --no-install-recommends python3-dev python3-pip swig

# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn

# Upgrade pip itself and install numpy and jupyter
RUN python -m pip install --upgrade pip
    # pip install numpy

# Copy GeNN into /opt
COPY  . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# Install GeNN and PyGeNN
RUN make install -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
# RUN python3 setup.py develop
RUN python setup.py bdist_wheel
RUN python setup.py bdist_wheel

# Copy the wheel to a new image for extraction
FROM scratch AS output
COPY --from=build $(GENN_PATH)/dist/*.whl /