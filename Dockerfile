ARG BASE=12.2.0-devel-ubuntu22.04
FROM nvidia/cuda:${BASE}

LABEL maintainer="J.C.Knight@sussex.ac.uk" \
    version="4.8.0" \
    org.opencontainers.image.documentation="https://genn-team.github.io/" \
    org.opencontainers.image.source="https://github.com/genn-team/genn" \
    org.opencontainers.image.title="GeNN Docker image"

# Update APT database and upgrade any outdated packages and install Python, pip and swig
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -yq --no-install-recommends python3-dev python3-pip swig gosu nano

# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn

# Set python3 to be the default version of python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip itself and install numpy and jupyter
RUN python -m pip install --upgrade pip && \
    python -m pip install numpy jupyter matplotlib

# Copy GeNN into /opt
COPY . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# Install GeNN and PyGeNN
RUN make install -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN python setup.py develop

# Start entrypoint
# **NOTE** in 'exec' mode shell arguments aren't expanded so can't use environment variables
ENTRYPOINT ["/opt/genn/bin/genn-docker-entrypoint.sh"]

# Default command will be to launch bash
CMD ["/bin/bash"]
