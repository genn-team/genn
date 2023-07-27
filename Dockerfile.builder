# syntax=docker/dockerfile:1

ARG CUDA=12.2
FROM sameli/manylinux2014_x86_64_cuda_${CUDA} AS build

LABEL maintainer="B.D.Evans@sussex.ac.uk" \
    org.opencontainers.image.documentation="https://genn-team.github.io/" \
    org.opencontainers.image.source="https://github.com/genn-team/genn" \
    org.opencontainers.image.title="PyGeNN wheel builder"

# install Python, pip and swig


# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn

# Set python3 to be the default version of python




# # Upgrade pip itself and install numpy and swig
# RUN python -m pip install --upgrade pip && \
#     python -m pip install numpy swig

# Copy GeNN into /opt
COPY . ${GENN_PATH}

# Use this as working directory
WORKDIR ${GENN_PATH}

# # Install GeNN and PyGeNN
# RUN make install -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
# RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
# # RUN python3 setup.py develop
# RUN python setup.py bdist_wheel
# RUN python setup.py bdist_wheel

# PLAT=manylinux2014_x86_64
# docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/travis/build-wheels.sh

RUN ./build-wheels.sh

# Copy the wheel to a new image for extraction
FROM scratch AS output
# TODO: Find a workaround for broken variable expansion
#ARG GENN_PATH
#COPY --from=build ${GENN_PATH}/dist/*.whl /
COPY --from=build /opt/genn/dist/*.whl /
