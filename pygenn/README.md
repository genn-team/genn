# A Python interface to GeNN
PyGeNN wraps the C++ GeNN APU using SWIG, allowing GeNN to be used either directly from Python or as a backend for higher-level Python APIs such as [PyNN](https://github.com/genn-team/pynn_genn). Currently PyGeNN only supports Windows and Mac and needs to be built from source.

### Installing PyGeNN from source on Linux or Mac OSX
 - Install swig from MacPorts or you Linux package manage
 - Download or clone GeNN and extract into your home directory
 - Ensure that the ``GENN_PATH`` environment variable to point to the GeNN directory.
 - From the GeNN directory, build GeNN as a dynamic library using ``make -f lib/GNUMakefileLibGeNN DYNAMIC=1 LIBGENN_PATH=pygenn/genn_wrapper/`` (you will need to add ``CPU_ONLY=1`` if you do not have an NVIDIA GPU)
 - On Mac OS X only, set your newly created library's name with ``install_name_tool -id "@loader_path/libgenn_DYNAMIC.dylib" pygenn/genn_wrapper/libgenn_DYNAMIC.dylib`` (you will need to replace ``libgenn_DYNAMIC`` with ``libgenn_CPU_ONLY_DYNAMIC`` if you do not have an NVIDIA GPU)
 - Install with setup tools using ``python setup.py develop`` command
