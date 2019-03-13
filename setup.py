#!/usr/bin/env python
import numpy as np
import os
import sys

from copy import deepcopy
from platform import system
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from six import iteritems

from generate_swig_interfaces import generateConfigs

# Get CUDA path, either default or from environment variable
cuda_path = (os.environ["CUDA_PATH"]
             if "CUDA_PATH" in os.environ
             else "/usr/local/cuda")

# Is CUDA installed?
cuda_installed = os.path.exists(cuda_path)

mac_os_x = system() == "Darwin"
linux = system() == "Linux"

genn_path = os.path.dirname(os.path.abspath(__file__))
numpy_path = os.path.join(os.path.dirname(np.__file__))

genn_wrapper_path = os.path.join(genn_path, "pygenn", "genn_wrapper")
genn_wrapper_include = os.path.join(genn_wrapper_path, "include")
genn_wrapper_swig = os.path.join(genn_wrapper_path, "swig")
genn_wrapper_generated = os.path.join(genn_wrapper_path, "generated")
genn_include = os.path.join(genn_path, "include", "genn")
genn_third_party_include = os.path.join(genn_path, "include", "genn", "third_party")

swig_opts = ["-c++", "-relativeimport", "-outdir", genn_wrapper_path, "-I" + genn_wrapper_include,
             "-I" + genn_wrapper_generated, "-I" + genn_wrapper_swig, "-I" + genn_include]

include_dirs = [genn_include, genn_third_party_include, genn_wrapper_include, genn_wrapper_generated,
                os.path.join(numpy_path, "core", "include")]

# If we are building for Python 3, add SWIG option (otherwise imports get broken)
# **YUCK** why doesn't setuptools do this automatically!?
if sys.version_info > (3, 0):
    swig_opts.append("-py3")

# By default link against libGeNN (relocatable version)
extra_compile_args = ["-std=c++11"]
library_dirs = [os.path.join(genn_path, "lib")]
libraries = ["genn_relocatable"]

# By default build single-threaded CPU backend
backends = [("genn_single_threaded_cpu_backend", "SingleThreadedCPU", {})]

# If CUDA was found, add backend configuration
if cuda_installed:
    backends.append(("genn_cuda_backend", "CUDA",
                     {"libraries": ["cuda", "cudart"],
                      "include_dirs": [os.path.join(cuda_path, "include")],
                      "library_dirs": [os.path.join(cuda_path, "lib") if mac_os_x else os.path.join(cuda_path, "lib64")]}))

# Build dictionary of kwargs to apply to all modules
extension_kwargs = {
    "swig_opts": swig_opts,
    "include_dirs": include_dirs,
    "libraries": libraries,
    "library_dirs": library_dirs + [genn_wrapper_path],
    "runtime_library_dirs": library_dirs,
    "extra_compile_args" : extra_compile_args}

# **HACK** on Mac OSX, "runtime_library_dirs" 
# doesn't actually work so add rpath manually instead
if mac_os_x:
    extension_kwargs["extra_link_args"] = ["-Wl,-rpath," + l
                                           for l in library_dirs]

# Before building extension, generate auto-generated parts of genn_wrapper
generateConfigs(genn_path, backends)

# Create main GeNNWrapper extension containing all of LibGeNN and base classes for implementing custom models in Python
genn_wrapper = Extension('_genn_wrapper', [
    "pygenn/genn_wrapper/generated/genn_wrapper.i",
    "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc",
    "pygenn/genn_wrapper/generated/initSparseConnectivitySnippetCustom.cc",
    "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc",
    "pygenn/genn_wrapper/generated/neuronModelsCustom.cc",
    "pygenn/genn_wrapper/generated/postsynapticModelsCustom.cc",
    "pygenn/genn_wrapper/generated/weightUpdateModelsCustom.cc"],
    **extension_kwargs)

# Create list of extension modules required to wrap other namespaces of libGeNN
ext_modules = [genn_wrapper,
               Extension('_Snippet', ["pygenn/genn_wrapper/swig/Snippet.i"], **extension_kwargs),
               Extension('_Models', ["pygenn/genn_wrapper/swig/Models.i"], **extension_kwargs),
               Extension('_StlContainers', ["pygenn/genn_wrapper/generated/StlContainers.i"], **extension_kwargs),
               Extension('_SharedLibraryModel', ["pygenn/genn_wrapper/generated/SharedLibraryModel.i"], **extension_kwargs),
               Extension('_InitVarSnippet', ["pygenn/genn_wrapper/generated/InitVarSnippet.i", "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc"], **extension_kwargs),
               Extension('_InitSparseConnectivitySnippet', ["pygenn/genn_wrapper/generated/InitSparseConnectivitySnippet.i", "pygenn/genn_wrapper/generated/initSparseConnectivitySnippetCustom.cc"], **extension_kwargs),
               Extension('_NeuronModels', ["pygenn/genn_wrapper/generated/NeuronModels.i", "pygenn/genn_wrapper/generated/neuronModelsCustom.cc"], **extension_kwargs),
               Extension('_PostsynapticModels', ["pygenn/genn_wrapper/generated/PostsynapticModels.i", "pygenn/genn_wrapper/generated/postsynapticModelsCustom.cc"], **extension_kwargs),
               Extension('_WeightUpdateModels', ["pygenn/genn_wrapper/generated/WeightUpdateModels.i", "pygenn/genn_wrapper/generated/weightUpdateModelsCustom.cc"], **extension_kwargs),
               Extension('_CurrentSourceModels', ["pygenn/genn_wrapper/generated/CurrentSourceModels.i", "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc"], **extension_kwargs)]

# Loop through namespaces of supported backends
for filename, namespace, kwargs in backends:
    # Take a copy of the standard extension kwargs
    backend_extension_kwargs = deepcopy(extension_kwargs)

    # Extend any settings specified by backend
    for n, v in iteritems(kwargs):
        backend_extension_kwargs[n].extend(v)

    # Add relocatable version of backend library to libraries
    # **NOTE** this is added BEFORE libGeNN as this library needs symbols FROM libGeNN
    backend_extension_kwargs["libraries"].insert(0, filename + "_relocatable")

    # Add backend include directory to both SWIG and C++ compiler options
    backend_include_dir = os.path.join(genn_path, "include", filename)
    backend_extension_kwargs["include_dirs"].append(backend_include_dir)
    backend_extension_kwargs["swig_opts"].append("-I" + backend_include_dir)

    # Add extension to list
    ext_modules.append(Extension("_" + namespace + "Backend", ["pygenn/genn_wrapper/generated/" + namespace + "Backend.i"],
                                 **backend_extension_kwargs))

setup(name = "pygenn",
      version = "0.2",
      packages = find_packages(),

      url="https://github.com/genn-team/genn",
      author="University of Sussex",
      description="Python interface to the GeNN simulator",
      ext_package="pygenn.genn_wrapper",
      ext_modules=ext_modules,

    # Requirements
    install_requires=["numpy>1.6, < 1.15", "six"],
    zip_safe=False,  # Partly for performance reasons
)
