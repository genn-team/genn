FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

ARG USERNAME="pygennuser"
ARG PYGENN_HOME="/root"

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -yq --no-install-recommends curl vim python3-dev python3-pip

RUN pip install swig numpy jupyter

ENV CUDA_PATH=/usr/local/cuda-11.5
ENV HOME=${PYGENN_HOME}
ENV USERNAME=${USERNAME}

WORKDIR ${HOME}

COPY . ${HOME}

RUN adduser --disabled-password --gecos "" $USERNAME && chown -R ${USERNAME}:${USERNAME} "/root"

RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${HOME}/pygenn/genn_wrapper/
RUN python3 setup.py develop

USER $USERNAME
