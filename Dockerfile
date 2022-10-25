FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Update APT database and upgrade any outdated packages
RUN apt-get update
#RUN apt-get update && \
#    apt-get upgrade -y

# Install Python, pip and swig
RUN apt-get install -yq --no-install-recommends python3-dev python3-pip swig


# Set CUDA environment variable
ENV CUDA_PATH=/usr/local/cuda-11.5

# From username, configure environment variable containing home directory
ENV USERNAME=pygenn
ENV HOME=/home/${USERNAME}

# Add local bin to path
ENV PATH="${PATH}:${HOME}/.local/bin"

# Add user non-interactively
RUN adduser --disabled-password --gecos "" ${USERNAME}

# Switch to user
USER $USERNAME

# Upgrade pip itself
RUN pip install --upgrade pip

# Install numpy and jupyter
RUN pip install numpy jupyter

# Copy GeNN into home directory
COPY  --chown=${USERNAME}:${USERNAME} . ${HOME}

# Set GeNN directory as current working directory
WORKDIR ${HOME}

# Install PyGeNN
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${HOME}/pygenn/genn_wrapper/ -j 8
RUN python3 setup.py develop --user

