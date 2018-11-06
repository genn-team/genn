#!/usr/bin/env python
import numpy as np
import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

from generate_swig_interfaces import generateConfigs

cpu_only = "CUDA_PATH" not in os.environ

genn_path = os.path.dirname(os.path.abspath(__file__))
numpy_path = os.path.join(os.path.dirname(np.__file__))

genn_wrapper_path = os.path.join(genn_path, "pygenn", "genn_wrapper")
genn_wrapper_include = os.path.join(genn_wrapper_path, "include")
genn_wrapper_swig = os.path.join(genn_wrapper_path, "swig")
genn_wrapper_generated = os.path.join(genn_wrapper_path, "generated")
genn_include = os.path.join(genn_path, "lib", "include")

swig_opts = ["-c++", "-outdir", genn_wrapper_path, "-I" + genn_wrapper_include,
             "-I" + genn_wrapper_generated, "-I" + genn_wrapper_swig, "-I" + genn_include]

include_dirs = [genn_include, genn_wrapper_include, genn_wrapper_generated,
                os.path.join(numpy_path, "core", "include")]

library_dirs = [genn_wrapper_path]

genn_wrapper_macros=[("GENERATOR_MAIN_HANDLED", None)]
extra_compile_args = ["-std=c++11"]

# If CUDA was found
if not cpu_only:
    # Get CUDA path
    cuda_path = os.path.join(os.environ["CUDA_PATH"])
    
    # Link against CUDA and CUDA version of GeNN
    libraries =["cuda", "cudart", "genn_DYNAMIC"]
    
    # Add CUDA include and library path
    include_dirs.append(os.path.join(cuda_path, "include"))
    library_dirs.append(os.path.join(cuda_path, "lib64"))
    
    # Add macro to point GeNN to NVCC compiler
    genn_wrapper_macros.append(("NVCC", "\"" + os.path.join(cuda_path, "bin", "nvcc") + "\""))
else:
    libraries = ["genn_CPU_ONLY_DYNAMIC"]
    genn_wrapper_macros.append(("CPU_ONLY","1"))
    swig_opts.append("-DCPU_ONLY")
    extra_compile_args.append("-DCPU_ONLY")

extension_kwargs = {
    "swig_opts": swig_opts,
    "include_dirs": include_dirs,
    "libraries": libraries,
    "library_dirs": library_dirs,
    "runtime_library_dirs": library_dirs,
    "extra_compile_args" : extra_compile_args}

# Before building extension, generate auto-generated parts of genn_wrapper
generateConfigs(genn_path)

genn_wrapper = Extension('_genn_wrapper', [
    "pygenn/genn_wrapper/generated/genn_wrapper.i",
    "lib/src/generateALL.cc", "lib/src/generateCPU.cc", 
    "lib/src/generateInit.cc", "lib/src/generateKernels.cc", 
    "lib/src/generateMPI.cc", "lib/src/generateRunner.cc",
    "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc",
    "pygenn/genn_wrapper/generated/initSparseConnectivitySnippetCustom.cc",
    "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc",
    "pygenn/genn_wrapper/generated/newNeuronModelsCustom.cc",
    "pygenn/genn_wrapper/generated/newPostsynapticModelsCustom.cc",
    "pygenn/genn_wrapper/generated/newWeightUpdateModelsCustom.cc"],
    define_macros=genn_wrapper_macros,
    **extension_kwargs)

setup(name = "pygenn",
      version = "0.1",
      packages = find_packages(),
      url="https://github.com/genn-team/genn",
      author="University of Sussex",
      description="Python interface to the GeNN simulator",
      ext_package="pygenn.genn_wrapper",
      ext_modules=[genn_wrapper,
                   Extension('_Snippet', ["pygenn/genn_wrapper/swig/Snippet.i"], **extension_kwargs),
                   Extension('_NewModels', ["pygenn/genn_wrapper/swig/NewModels.i"], **extension_kwargs),
                   Extension('_GeNNPreferences', ["pygenn/genn_wrapper/swig/GeNNPreferences.i"], **extension_kwargs),
                   Extension('_StlContainers', ["pygenn/genn_wrapper/generated/StlContainers.i"], **extension_kwargs),
                   Extension('_SharedLibraryModel', ["pygenn/genn_wrapper/generated/SharedLibraryModel.i"], **extension_kwargs),
                   Extension('_InitVarSnippet', ["pygenn/genn_wrapper/generated/InitVarSnippet.i", "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc"], **extension_kwargs),
                   Extension('_InitSparseConnectivitySnippet', ["pygenn/genn_wrapper/generated/InitSparseConnectivitySnippet.i", "pygenn/genn_wrapper/generated/initSparseConnectivitySnippetCustom.cc"], **extension_kwargs),
                   Extension('_NeuronModels', ["pygenn/genn_wrapper/generated/NeuronModels.i", "pygenn/genn_wrapper/generated/newNeuronModelsCustom.cc"], **extension_kwargs),
                   Extension('_PostsynapticModels', ["pygenn/genn_wrapper/generated/PostsynapticModels.i", "pygenn/genn_wrapper/generated/newPostsynapticModelsCustom.cc"], **extension_kwargs),
                   Extension('_WeightUpdateModels', ["pygenn/genn_wrapper/generated/WeightUpdateModels.i", "pygenn/genn_wrapper/generated/newWeightUpdateModelsCustom.cc"], **extension_kwargs),
                   Extension('_CurrentSourceModels', ["pygenn/genn_wrapper/generated/CurrentSourceModels.i", "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc"], **extension_kwargs)])
