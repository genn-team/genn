#!/usr/bin/env python
import numpy as np
import os
import sys

from platform import system
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

from generate_swig_interfaces import generateConfigs

# Get CUDA path, either default or from environment variable
cuda_path = (os.environ["CUDA_PATH"]
             if "CUDA_PATH" in os.environ
             else "/usr/local/cuda")

# Use CPU ONLY mode if CUDA path doesn't exist
cpu_only = True#not os.path.exists(cuda_path)

mac_os_x = system() == "Darwin"
linux = system() == "Linux"

genn_path = os.path.dirname(os.path.abspath(__file__))
numpy_path = os.path.join(os.path.dirname(np.__file__))

genn_wrapper_path = os.path.join(genn_path, "pygenn", "genn_wrapper")
genn_wrapper_include = os.path.join(genn_wrapper_path, "include")
genn_wrapper_swig = os.path.join(genn_wrapper_path, "swig")
genn_wrapper_generated = os.path.join(genn_wrapper_path, "generated")
genn_include = os.path.join(genn_path, "lib", "include")

swig_opts = ["-c++", "-relativeimport", "-outdir", genn_wrapper_path, "-I" + genn_wrapper_include,
             "-I" + genn_wrapper_generated, "-I" + genn_wrapper_swig, "-I" + genn_include]

include_dirs = [genn_include, genn_wrapper_include, genn_wrapper_generated,
                os.path.join(numpy_path, "core", "include")]

# If we are building for Python 3, add SWIG option (otherwise imports get broken)
# **YUCK** why doesn't setuptools do this automatically!?
if sys.version_info > (3, 0):
    swig_opts.append("-py3")

library_dirs = []
genn_wrapper_macros=[("GENERATOR_MAIN_HANDLED", None)]
extra_compile_args = ["-std=c++11"]

# If CUDA was found
if not cpu_only:
    # Link against CUDA and CUDA version of GeNN
    libraries =["cuda", "cudart", "genn_DYNAMIC"]
    
    # Add CUDA include and library path
    include_dirs.append(os.path.join(cuda_path, "include"))
    if mac_os_x:
        library_dirs.append(os.path.join(cuda_path, "lib"))
    else:
        library_dirs.append(os.path.join(cuda_path, "lib64"))

    # Add macro to point GeNN to NVCC compiler
    genn_wrapper_macros.append(("NVCC", "\"" + os.path.join(cuda_path, "bin", "nvcc") + "\""))

    # Make sure LibGeNN gets packages
    package_data = ["genn_wrapper/*genn_DYNAMIC.*"]
# Otherwise
else:
    # Link against CPU-only version of GeNN
    libraries = ["genn_CPU_ONLY_DYNAMIC"]

    # Set CPU_ONLY macro everywhere
    genn_wrapper_macros.append(("CPU_ONLY","1"))
    swig_opts.append("-DCPU_ONLY")
    extra_compile_args.append("-DCPU_ONLY")

    # Make sure LibGeNN gets packages
    package_data = ["genn_wrapper/*genn_CPU_ONLY_DYNAMIC.*"]

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
# Conversely, on Linux, we want to add extension directory i.e. $ORIGIN to runtime
#directories so ligGeNN can be found wherever package is installed
elif linux:
    extension_kwargs["runtime_library_dirs"].append("$ORIGIN")

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



# We do some trickery to assure SWIG is always run before installing the generated files.
# http://stackoverflow.com/questions/12491328/python-distutils-not-include-the-swig-generated-module
class CustomBuild(build):
    def run(self):
        self.run_command("build_ext")
        build.run(self)

class CustomInstall(install):
    def run(self):
        self.run_command("build_ext")
        self.do_egg_install()

setup(name = "pygenn",
      version = "0.1",
      cmdclass={"build": CustomBuild, "install": CustomInstall},
      packages = find_packages(),
      package_data={"pygenn": package_data},

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
                   Extension('_CurrentSourceModels', ["pygenn/genn_wrapper/generated/CurrentSourceModels.i", "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc"], **extension_kwargs)],

    # Requirements
    install_requires=["numpy>1.6, < 1.15", "six"],
    zip_safe=False,  # Partly for performance reasons
)
