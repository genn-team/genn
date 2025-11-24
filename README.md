
[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN%2Fgenn%2Fmaster)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/job/genn/job/master/)[![codecov.io](https://codecov.io/github/genn-team/genn/coverage.svg?branch=master)](https://codecov.io/github/genn-team/genn?branch=master) [![DOI](https://zenodo.org/badge/24633934.svg)](https://zenodo.org/badge/latestdoi/24633934)
[![Dockerhub](https://img.shields.io/badge/dockerhub-images-orange.svg?logo=docker)](https://hub.docker.com/repository/docker/gennteam/genn) [![Neuromorphic Computing](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/neuromorphic-computing/)

# GPU-enhanced Neuronal Networks (GeNN)

GeNN is a GPU-enhanced Neuronal Network simulation environment based on code generation for NVIDIA CUDA and AMD HIP.

## Installation

### Pre-installation

1.  Install the C++ compiler on the machine, if not already present.
    For Windows, Visual Studio 2019 or above is required. The Microsoft Visual Studio 
    Community Edition can be downloaded from
    https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx.
    When installing Visual Studio, one should select the 'Desktop 
    development with C++' configuration. On Linux, the GNU Compiler 
    Collection (GCC) 7.5 or above is required. This can be obtained from your
    Linux distribution repository, for example on Ubuntu by running ``sudo apt-get install g++``, 
    or alternatively from https://gcc.gnu.org/index.html.
2.  If your machine has an NVIDIA GPU and you haven't installed CUDA already, 
    obtain a fresh installation of the NVIDIA CUDA toolkit from
    https://developer.nvidia.com/cuda-downloads
    Be sure to pick CUDA and C++ compiler versions which are compatible
    with each other. The latest C++ compiler need not necessarily be
    compatible with the latest CUDA toolkit. Similarly, if your machine 
    has an AMD GPU and you haven't installed HIP yet, follow the instructions at
    https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html.
3.  GeNN uses the ``CUDA_PATH`` environment variable to determine which 
    version of CUDA to build against. On Windows, this is set automatically when 
    installing CUDA. However, if you choose, you can verify which version is 
    selected by running ``echo %CUDA_PATH%`` in a command prompt.
    However, on Linux, you need to set ``CUDA_PATH`` manually with:
    ``export CUDA_PATH=/usr/local/cuda``
    assuming CUDA is installed in /usr/local/cuda (the standard location 
    on Ubuntu Linux). Similarly, if you are using HIP, you need to set the 
    ``HIP_PATH`` variable manually and also specify your platform with either
    ``export HIP_PLATFORM='nvidia'`` if you wish to use HIP with an NVIDIA GPU
    or ``export HIP_PLATFORM='amd'`` if you wish to use an AMD GPU.
    To make any of these changes persistent, these commands should be added to your login 
    script (e.g. ``.profile`` or ``.bashrc``).
4.  On Linux, install the development version of libffi. For example, on Ubuntu you can do this
    by running ``sudo apt-get install libffi-dev``.

### Installation using pip
The easiest way to install GeNN is directly from github using pip. 
First of all make sure pip is up to date using :
``pip install -U pip``
Then, to install the latest development version you can use:
``pip install https://github.com/genn-team/genn/archive/refs/heads/master.zip`` or, to install the 5.3.0 release, you can use: 
``pip install https://github.com/genn-team/genn/archive/refs/tags/5.3.0.zip``.

### Creating an editable install with pip
If you want to develop GeNN yourself or run userprojects from the GeNN repository, it is helpful to create an 'editable' install. 
The easiest way to do this to first 'clone' GeNN from github using ``git clone https://github.com/genn-team/genn.git``.
Then, navigate to the GeNN directory and install using ``pip install -e .``. If you wish to install the additional dependencies needed 
to run the userprojects, you can do so using ``pip install -e .[userproject]``.

### Building with setup.py (LEGACY)
Although it is not recommended, in order to build special development versions you sometimes need to install the old fashioned way!
1.  Manually install PyGeNN's build dependencies using pip i.e. ``pip install pybind11 psutil pkgconfig setuptools>=61``.
2.  Clone GeNN using git i.e. using ``git clone https://github.com/genn-team/genn.git``
3.  From the GeNN directory, build PyGeNN using ``python setup.py develop``. 
    You can build a debug version of GeNN with ``python setup.py build_ext --debug develop``.

## Docker
You can also use GeNN through our CUDA-enabled docker container which comes with GeNN pre-installed.
To work with such CUDA-enabled containers, you need to first install CUDA on your host system as described above and then install docker and the NVIDIA Container Toolkit as described in https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker.
You can then build the GeNN container yourself or download it from Dockerhub.

### Building the container
The following command can be used from the GeNN source directory to build the GeNN container:
```bash
make docker-build
```

This builds a container tagged as ``genn:latest`` so, to use this container rather than downloading the prebuild one from dockerhub, just replace ``gennteam/genn:latest`` with ``genn:latest`` in the following instructions.
By default, the container image is based off the Ubuntu 20.04 image with CUDA 11.5 provided by NVIDIA but, if you want to use a different base image, for example to use the container on a machine with an older version of CUDA, you can invoke ``docker build`` directly and specify a different tag (listed on https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md)  via the ``BASE`` build argument. For example to build using CUDA 11.3 you could run:
```bash
docker build  --build-arg BASE=11.3.0-devel-ubuntu20.04 -t genn:latest_cuda_11_3 .
```

### Interactive mode
If you wish to use GeNN or PyGeNN interactively, you can launch a bash shell in the GeNN container using the following command:
```bash
docker run -it --gpus=all gennteam/genn:latest
```
You can also provide a final argument to launch a different executable e.g. ``/bin/sh`` to launch a dash shell.
**NOTE** PyGeNN is installed in the system Python 3 environment, the interpreter for which is launched with ``python3`` (rather than just ``python``) on Ubuntu 20.04.

### Accessing your files
When using the GeNN container you often want to access files on your host system.
This can be easily achieved by using the ``-v`` option to mount a local directory into the container. For example:
```bash
docker run -it --gpus=all -v $HOME:/local_home gennteam/genn:latest
```
mounts the local user's home directory into ``/local_home`` within the container.
However, all of the commands provided by the GeNN container operate using a non-elevated, internal user called 'genn' who, by default, won't have the correct permissions to create files in volumes mounted into the container.
This can be resolved by setting the ``LOCAL_USER_ID`` and ``LOCAL_GROUP_ID`` environment variables when running the container like:
```bash
docker run -it --gpus=all -e LOCAL_USER_ID=`id -u $USER` -e LOCAL_GROUP_ID=`id -g $USER` -v $HOME:/local_home gennteam/genn:latest
```
which will ensure that that 'genn' user has the same UID and GID as the local user, meaning that they will have the same permissions to access the files mounted into ``/local_home``. 

### Running Jupyter Notebooks
A Jupyter Notebook environment running in the container can be launched using the ``notebook`` command. Typically, you would combine this with the ``-p 8080:8080`` option to 'publish' port 8080, allowing the notebook server to be accessed on the host. By default, notebooks are created in the home directory of the 'genn' user inside the container. However, to create notebooks which persist beyond the lifetime of the container, the notebook command needs to be combined with the options discussed previously. For example:
```bash
docker run --gpus=all -p 8080:8080 -e LOCAL_USER_ID=`id -u $USER` -e LOCAL_GROUP_ID=`id -g $USER` -v $HOME:/local_home gennteam/genn:latest notebook /local_home
```
will create notebooks in the current users home directory.

### Running PyGeNN scripts
Assuming they have no additional dependencies, PyGeNN scripts can be run directly using the container with the ``script`` command. As scripts are likely to be located outside of the container, the script command is often combined with the options discussed previously. For example, to run a script called ``test.py`` in your home directory, the script command could be invoked with:
```bash
docker run --gpus=all -e LOCAL_USER_ID=`id -u $USER` -e LOCAL_GROUP_ID=`id -g $USER` -v $HOME:/local_home gennteam/genn:latest script /local_home/test.py
```

## Usage

### Sample projects

At the moment, the following Python example projects are provided with GeNN:

- Cortical microcircuit model \([Potjans et al. 2014][@Potjans2014]\)
- SuperSpike model \([Zenke et al. 2018][@Zenke2018]\)
- MNIST classifier using an insect-inspired mushroom body model

In order to get a quick start and run one of the the provided example models, navigate to the userproject directory, and run the python script with ``--help`` to see what options are available.

For more details on how to use GeNN, please see [documentation](http://genn-team.github.io/genn/).

[@Potjans2014]: https://doi.org/10.1093/cercor/bhs358 "Potjans, T. C., & Diesmann, M. The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model. Cerebral Cortex, 24(3), 785–806 (2014)"
[@Zenke2018]: https://doi.org/10.1162/neco_a_01086  "Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, 30(6), 1514–1541."
