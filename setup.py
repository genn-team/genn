#!/usr/bin/env python
import numpy as np
import os
import sys

from copy import deepcopy
from platform import system
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from shutil import copytree, rmtree
from generate_swig_interfaces import generateConfigs

# Determine is this is a debug build
# **YUCK** this is not a great test
debug_build = "--debug" in sys.argv

# Get CUDA path from environment variable - setting this up is a required CUDA post-install step
cuda_path = os.environ.get("CUDA_PATH")

# Is CUDA installed?
cuda_installed = cuda_path is not None and os.path.exists(cuda_path)

# Get OpenCL path from environment variable
opencl_path = os.environ.get("OPENCL_PATH")

# Is OpenCL installed
opencl_installed = opencl_path is not None and os.path.exists(opencl_path)

mac_os_x = system() == "Darwin"
linux = system() == "Linux"
windows = system() == "Windows"

# Determine correct suffix for GeNN libraries
if windows:
    genn_lib_suffix = "_Debug_DLL" if debug_build else "_Release_DLL"
else:
    genn_lib_suffix = "_dynamic_debug" if debug_build else "_dynamic"

genn_path = os.path.dirname(os.path.abspath(__file__))
numpy_path = os.path.join(os.path.dirname(np.__file__))

pygenn_path = os.path.join(genn_path, "pygenn")
genn_wrapper_path = os.path.join(pygenn_path, "genn_wrapper")
genn_wrapper_include = os.path.join(genn_wrapper_path, "include")
genn_wrapper_swig = os.path.join(genn_wrapper_path, "swig")
genn_wrapper_generated = os.path.join(genn_wrapper_path, "generated")
genn_include = os.path.join(genn_path, "include", "genn", "genn")
genn_third_party_include = os.path.join(genn_path, "include", "genn", "third_party")
genn_share = os.path.join(genn_path, "share", "genn")
pygenn_share = os.path.join(pygenn_path, "share")

swig_opts = ["-c++", "-relativeimport", "-outdir", genn_wrapper_path, "-I" + genn_wrapper_include,
             "-I" + genn_wrapper_generated, "-I" + genn_wrapper_swig]

include_dirs = [genn_wrapper_include, genn_wrapper_generated,
                os.path.join(numpy_path, "core", "include")]

# If we are building for Python 3, add SWIG option (otherwise imports get broken)
# **YUCK** why doesn't setuptools do this automatically!?
if sys.version_info > (3, 0):
    swig_opts.append("-py3")

# Build dictionary of kwargs to apply to all modules
extension_kwargs = {
    "swig_opts": swig_opts,
    "include_dirs": include_dirs,
    "library_dirs": [genn_wrapper_path],
    "extra_compile_args" : ["/wd\"4251\"", "-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX"] if windows else ["-std=c++11"],
    "extra_link_args": []}

# Always package LibGeNN
package_data = ["genn_wrapper/genn" + genn_lib_suffix + ".*"] if windows else ["genn_wrapper/libgenn" + genn_lib_suffix + ".*"]

# Copy dictionary and add libGeNN to apply to all modules that link against GeNN
genn_extension_kwargs = deepcopy(extension_kwargs)
genn_extension_kwargs["libraries"] = ["genn" + genn_lib_suffix]

genn_extension_kwargs["include_dirs"].extend([genn_include, genn_third_party_include])
genn_extension_kwargs["swig_opts"].extend(["-I" + genn_include, "-I" + genn_third_party_include])
genn_extension_kwargs["define_macros"] = [("LINKING_GENN_DLL", "1"), ("LINKING_BACKEND_DLL", "1")]

# On Linux, we want to add extension directory i.e. $ORIGIN to runtime
# directories so libGeNN and backends can be found wherever package is installed
if linux:
    genn_extension_kwargs["runtime_library_dirs"] = ["$ORIGIN"]

# By default build single-threaded CPU backend
backends = [("single_threaded_cpu", "SingleThreadedCPU", {})]

# If CUDA was found, add backend configuration
if cuda_installed:
    # Get CUDA library directory
    if mac_os_x:
        cuda_library_dir = os.path.join(cuda_path, "lib")
    elif windows:
        cuda_library_dir = os.path.join(cuda_path, "lib", "x64")
    else:
        cuda_library_dir = os.path.join(cuda_path, "lib64")

    # Add backend
    # **NOTE** on Mac OS X, a)runtime_library_dirs doesn't work b)setting rpath is required to find CUDA
    backends.append(("cuda", "CUDA",
                     {"libraries": ["cuda", "cudart"],
                      "include_dirs": [os.path.join(cuda_path, "include")],
                      "library_dirs": [cuda_library_dir],
                      "extra_link_args": ["-Wl,-rpath," + cuda_library_dir] if mac_os_x else []}))

# If OpenCL was found, add backend configuration
if opencl_installed:
    # Get OpenCL library directory
    if mac_os_x:
        raise NotImplementedError("Mac not currently supported")
    elif windows:
        opencl_library_dir = os.path.join(opencl_path, "lib", "x64")
    else:
        opencl_library_dir = os.path.join(opencl_path, "lib64")
    
    # Add backend
    # **NOTE** on Mac OS X, a)runtime_library_dirs doesn't work b)setting rpath is required to find CUDA
    backends.append(("opencl", "OpenCL",
                     {"libraries": ["OpenCL"],
                      "include_dirs": [os.path.join(opencl_path, "include")],
                      "library_dirs": [opencl_library_dir],
                      "extra_link_args": ["-Wl,-rpath," + opencl_library_dir] if mac_os_x else [],
                      "extra_compile_args": ["-DCL_HPP_TARGET_OPENCL_VERSION=120", "-DCL_HPP_MINIMUM_OPENCL_VERSION=120"]}))

# Before building extension, generate auto-generated parts of genn_wrapper
generateConfigs(genn_path, backends)

# Copy GeNN 'share' tree into pygenn and add all files to pacakge
# **THINK** this could be done on a per-backend basis
rmtree(pygenn_share, ignore_errors=True)
copytree(genn_share, pygenn_share)
for root, _, filenames in os.walk(pygenn_share):
    for f in filenames:
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path):
            package_data.append(f_path)

# Create list of extension modules required to wrap utilities and various libGeNN namespaces
ext_modules = [Extension('_StlContainers', ["pygenn/genn_wrapper/generated/StlContainers.i"], **extension_kwargs),
               Extension('_SharedLibraryModelNumpy', ["pygenn/genn_wrapper/generated/SharedLibraryModelNumpy.i"], **extension_kwargs),
               Extension('_genn_wrapper', ["pygenn/genn_wrapper/generated/genn_wrapper.i"], **genn_extension_kwargs),
               Extension('_Snippet', ["pygenn/genn_wrapper/swig/Snippet.i"], **genn_extension_kwargs),
               Extension('_Models', ["pygenn/genn_wrapper/swig/Models.i"], **genn_extension_kwargs),
               Extension('_InitVarSnippet', ["pygenn/genn_wrapper/generated/InitVarSnippet.i", "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc"], **genn_extension_kwargs),
               Extension('_InitSparseConnectivitySnippet', ["pygenn/genn_wrapper/generated/InitSparseConnectivitySnippet.i", "pygenn/genn_wrapper/generated/initSparseConnectivitySnippetCustom.cc"], **genn_extension_kwargs),
               Extension('_NeuronModels', ["pygenn/genn_wrapper/generated/NeuronModels.i", "pygenn/genn_wrapper/generated/neuronModelsCustom.cc"], **genn_extension_kwargs),
               Extension('_PostsynapticModels', ["pygenn/genn_wrapper/generated/PostsynapticModels.i", "pygenn/genn_wrapper/generated/postsynapticModelsCustom.cc"], **genn_extension_kwargs),
               Extension('_WeightUpdateModels', ["pygenn/genn_wrapper/generated/WeightUpdateModels.i", "pygenn/genn_wrapper/generated/weightUpdateModelsCustom.cc"], **genn_extension_kwargs),
               Extension('_CurrentSourceModels', ["pygenn/genn_wrapper/generated/CurrentSourceModels.i", "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc"], **genn_extension_kwargs),
               Extension('_CustomUpdateModels', ["pygenn/genn_wrapper/generated/CustomUpdateModels.i", "pygenn/genn_wrapper/generated/customUpdateModelsCustom.cc"], **genn_extension_kwargs)]

# Loop through namespaces of supported backends
for filename, namespace, kwargs in backends:
    # Take a copy of the standard extension kwargs
    backend_extension_kwargs = deepcopy(genn_extension_kwargs)

    # Extend any settings specified by backend
    for n, v in kwargs.items():
        backend_extension_kwargs[n].extend(v)

    # Add relocatable version of backend library to libraries
    # **NOTE** this is added BEFORE libGeNN as this library needs symbols FROM libGeNN
    if windows:
        package_data.append("genn_wrapper/genn_" + filename + "_backend" + genn_lib_suffix + ".*")
    else:
        package_data.append("genn_wrapper/libgenn_" + filename + "_backend" + genn_lib_suffix + ".*")

    # Add backend include directory to both SWIG and C++ compiler options
    backend_include_dir = os.path.join(genn_path, "include", "genn", "backends", filename)
    backend_extension_kwargs["libraries"].insert(0, "genn_" + filename + "_backend" + genn_lib_suffix)
    backend_extension_kwargs["include_dirs"].append(backend_include_dir)
    backend_extension_kwargs["swig_opts"].append("-I" + backend_include_dir)

    # Add extension to list
    ext_modules.append(Extension("_" + namespace + "Backend", ["pygenn/genn_wrapper/generated/" + namespace + "Backend.i"],
                                 **backend_extension_kwargs))

setup(name = "pygenn",
      version = "0.4.6",
      packages = find_packages(),
      package_data={"pygenn": package_data},

      url="https://github.com/genn-team/genn",
      author="University of Sussex",
      description="Python interface to the GeNN simulator",
      ext_package="pygenn.genn_wrapper",
      ext_modules=ext_modules,

      # Requirements
      install_requires=["numpy>=1.17", "six", "deprecated", "psutil"],
      zip_safe=False,  # Partly for performance reasons
)
