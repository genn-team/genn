# syntax=docker/dockerfile:1

ARG CUDA=12.2
FROM sameli/manylinux2014_x86_64_cuda_${CUDA} AS build

LABEL maintainer="B.D.Evans@sussex.ac.uk" \
    org.opencontainers.image.documentation="https://genn-team.github.io/" \
    org.opencontainers.image.source="https://github.com/genn-team/genn" \
    org.opencontainers.image.title="PyGeNN wheel builder"

# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn

# Copy GeNN into /opt
COPY . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

RUN ./build-wheels.sh

# Copy the wheel to a new image for extraction
FROM scratch AS output
# TODO: Find a workaround for broken variable expansion
#ARG GENN_PATH
#COPY --from=build ${GENN_PATH}/dist/*.whl /
COPY --from=build /opt/genn/dist/*.whl /
