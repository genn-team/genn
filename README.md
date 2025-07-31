
[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN%2Fgenn%2Fmaster)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/job/genn/job/master/)[![codecov.io](https://codecov.io/github/genn-team/genn/coverage.svg?branch=master)](https://codecov.io/github/genn-team/genn?branch=master) [![DOI](https://zenodo.org/badge/24633934.svg)](https://zenodo.org/badge/latestdoi/24633934)
[![Dockerhub](https://img.shields.io/badge/dockerhub-images-orange.svg?logo=docker)](https://hub.docker.com/repository/docker/gennteam/genn) [![Neuromorphic Computing](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/neuromorphic-computing/)

# GPU-enhanced Neuronal Networks (GeNN)

GeNN is a GPU-enhanced Neuronal Network simulation environment based on code generation for Nvidia CUDA.

## Installation

You can download GeNN either as a zip file of a stable release, checkout the development
version using the Git version control system or use our Docker container.

### Downloading a release
Point your browser to https://github.com/genn-team/genn/releases
and download a release from the list by clicking the relevant source
code button. After downloading continue to install GeNN as described in the [GitHub installing section](#installing-genn) below.

### Obtaining a Git snapshot

If it is not yet installed on your system, download and install Git
(http://git-scm.com/). Then clone the GeNN repository from Github
```bash
git clone https://github.com/genn-team/genn.git
```
The github url of GeNN in the command above can be copied from the
HTTPS clone URL displayed on the GeNN Github page (https://github.com/genn-team/genn).

This will clone the entire repository, including all open branches.
By default git will check out the master branch which contains the
source version upon which the next release will be based. There are other 
branches in the repository that are used for specific development 
purposes and are opened and closed without warning.

### Installing GeNN

In future we plan on providing binary builds of GeNN via conda. However, for now, GeNN
needs to be installed from source.

#### Pre-installation

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
    compatible with the latest CUDA toolkit.
3.  GeNN uses the ``CUDA_PATH`` environment variable to determine which 
    version of CUDA to build against. On Windows, this is set automatically when 
    installing CUDA. However, if you choose, you can verify which version is 
    selected by running ``echo $CUDA_PATH`` in a command prompt.
    However, on Linux, you need to set ``CUDA_PATH`` manually with:
    ``export CUDA_PATH=/usr/local/cuda``
    assuming CUDA is installed in /usr/local/cuda (the standard location 
    on Ubuntu Linux). Again, to make this change persistent, this can
    be added to your login script (e.g. ``.profile`` or ``.bashrc``)
4.  Either download the latest release of GeNN and extract into your 
    home directory or clone using git from https://github.com/genn-team/genn
5.  On Linux, install the development version of libffi. For example, on Ubuntu you can do this
    by running ``sudo apt-get install libffi-dev``.
6.  Install the pybind11, psutil and numpy packages with pip i.e. ``pip install pybind11 psutil numpy``.


#### Building with setup.py
From the GeNN directory, the GeNN libraries and python package can be built
with ``python setup.py install``. If you wish to create an editable install
(most useful if you are intending to modify GeNN yourself) you can also used
``python setup.py develop``. On Linux (or Windows if you have a debug version
of the python libraries installed) you can build a debug version of GeNN with
``python setup.py build_ext --debug develop``.

#### Building with pip
From the GeNN directory, the GeNN libraries and python package can be built
with ``pip install .``. If you wish to create an editable install
(most useful if you are intending to modify GeNN yourself) you can also used
``pip install --editable .``.

### Docker
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
