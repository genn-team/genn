.. py:currentmodule:: pygenn

============
Installation
============
In future we plan on providing binary builds of GeNN via conda. However, for now, GeNN
needs to be installed from source.

----------------
Pre-installation
----------------
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
    https://developer.nvidia.com/cuda-downloads.
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

----------------------
Installation using pip
----------------------
The easiest way to install GeNN is directly from github using pip. 
First of all make sure pip is up to date using :
``pip install -U pip``
Then, to install the latest development version you can use:
``pip install https://github.com/genn-team/genn/archive/refs/heads/master.zip`` or, to install the 5.3.0 release, you can use: 
``pip install https://github.com/genn-team/genn/archive/refs/tags/5.3.0.zip``.

-------------------------------------
Creating an editable install with pip
-------------------------------------
If you want to develop GeNN yourself or run userprojects from the GeNN repository, it is helpful to create an 'editable' install. 
The easiest way to do this to first 'clone' GeNN from github using ``git clone https://github.com/genn-team/genn.git``.
Then, navigate to the GeNN directory and install using ``pip install -e .``. If you wish to install the additional dependencies needed 
to run the userprojects, you can do so using ``pip install -e .[userproject]``.

-------------------------------
Building with setup.py (LEGACY)
-------------------------------
Although it is not recommended, in order to build special development versions you sometimes need to install the old fashioned way!

1.  Manually install PyGeNN's build dependencies using pip i.e. ``pip install pybind11 psutil pkgconfig setuptools>=61``.
2.  Clone GeNN using git i.e. using ``git clone https://github.com/genn-team/genn.git``
3.  From the GeNN directory, build PyGeNN using ``python setup.py develop``. 
    You can build a debug version of GeNN with ``python setup.py build_ext --debug develop``.
