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
    compatible with the latest CUDA toolkit.
3.  GeNN uses the ``CUDA_PATH`` environment variable to determine which 
    version of CUDA to build against. On Windows, this is set automatically when 
    installing CUDA. However, if you choose, you can verify which version is 
    selected by running ``echo %CUDA_PATH`` in a command prompt.
    However, on Linux, you need to set ``CUDA_PATH`` manually with:
    ``export CUDA_PATH=/usr/local/cuda``
    assuming CUDA is installed in /usr/local/cuda (the standard location 
    on Ubuntu Linux). Again, to make this change persistent, this can
    be added to your login script (e.g. ``.profile``, ``.bash_profile`` or ``.bashrc``)
4.  Either download the latest release of GeNN and extract it into your 
    home directory or clone using git from https://github.com/genn-team/genn
5.  On Linux, install the development version of libffi. For example, on Ubuntu you can do this
    by running ``sudo apt-get install libffi-dev``.
6.  Install the pybind11, psutil and numpy packages with pip, i.e. ``pip install pybind11 psutil numpy``.

----------------------
Building with setup.py
----------------------
From the GeNN directory, the GeNN libraries and python package can be built
with ``python setup.py install``. If you wish to create an editable install
(most useful if you are intending to modify GeNN yourself) you can also use
``python setup.py develop``. On Linux (or Windows if you have a debug version
of the python libraries installed) you can build a debug version of GeNN with
``python setup.py build_ext --debug develop``.
 
-----------------
Building with pip
-----------------
From the GeNN directory, the GeNN libraries and python package can be built
with ``pip install .``. If you wish to create an editable install
(most useful if you are intending to modify GeNN yourself) you can also use
``pip install --editable .``.
