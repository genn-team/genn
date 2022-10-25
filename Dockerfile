FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Update APT database and upgrade any outdated packages
RUN apt-get update
#RUN apt-get update && \
#    apt-get upgrade -y

# Install Python, pip and swig
RUN apt-get install -yq --no-install-recommends python3-dev python3-pip swig

# Set CUDA environment variable
ENV CUDA_PATH=/usr/local/cuda-11.5

# Upgrade pip itself
RUN pip install --upgrade pip

# Install numpy and jupyter
RUN pip install numpy jupyter

# Copy GeNN into /tmp
COPY  . /tmp/genn

# Use this as working directory
WORKDIR /tmp/genn

# Install PyGeNN
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=/tmp/genn/pygenn/genn_wrapper/ -j 8
RUN python3 setup.py install 
RUN python3 setup.py install 

# Add non-elevated user non-interactively
ENV USERNAME=pygenn
RUN adduser --disabled-password --gecos "" ${USERNAME}

# Switch to user
USER ${USERNAME}
