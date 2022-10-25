FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Update APT database and upgrade any outdated packages
RUN apt-get update
#RUN apt-get update && \
#    apt-get upgrade -y

# Install Python, pip and swig
RUN apt-get install -yq --no-install-recommends python3-dev python3-pip swig gosu nano

# Set CUDA environment variable
ENV CUDA_PATH=/usr/local/cuda-11.5

ENV GENN_PATH=/opt/genn

# Upgrade pip itself
RUN pip install --upgrade pip

# Install numpy and jupyter
RUN pip install numpy jupyter matplotlib

# Copy GeNN into /opt
COPY  . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# Install PyGeNN
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN python3 setup.py develop

# Start entrypoint
# **NOTE** in 'exec' mode shell arguments aren't expanded so can't use environment variables
ENTRYPOINT ["/opt/genn/bin/docker-entrypoint.sh"]
