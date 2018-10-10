#!/usr/bin/env python
import numpy as np
import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

from generate_swig_interfaces import generateConfigs

'''
class MakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext_make(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_make(ext)
        super().run()

    def build_make(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))
        '''
cuda_path = os.path.join(os.environ["CUDA_PATH"])
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
                os.path.join(cuda_path, "include"),
                os.path.join(numpy_path, "core", "include")]

libraries =["cuda", "cudart", "genn_DYNAMIC"]

library_dirs = [os.path.join(cuda_path, "lib64"), genn_wrapper_path]

extension_kwargs = {
    "swig_opts": swig_opts,
    "include_dirs": include_dirs,
    "libraries": libraries,
    "library_dirs": library_dirs,
    "runtime_library_dirs": library_dirs,
    "extra_compile_args" : ["-std=c++11"]}

# Before building extension, generate auto-generated parts of genn_wrapper
generateConfigs(genn_path)

genn_wrapper = Extension('_genn_wrapper', [
    "pygenn/genn_wrapper/generated/genn_wrapper.i",
    "lib/src/generateALL.cc", "lib/src/generateCPU.cc", 
    "lib/src/generateInit.cc", "lib/src/generateKernels.cc", 
    "lib/src/generateMPI.cc", "lib/src/generateRunner.cc",
    "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc",
    "pygenn/genn_wrapper/generated/initVarSnippetCustom.cc",
    "pygenn/genn_wrapper/generated/newNeuronModelsCustom.cc",
    "pygenn/genn_wrapper/generated/newPostsynapticModelsCustom.cc",
    "pygenn/genn_wrapper/generated/newWeightUpdateModelsCustom.cc"],
    define_macros=[("GENERATOR_MAIN_HANDLED", None), ("NVCC", "\"" + os.path.join(cuda_path, "bin", "nvcc") + "\"")],
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
                   Extension('_NeuronModels', ["pygenn/genn_wrapper/generated/NeuronModels.i", "pygenn/genn_wrapper/generated/newNeuronModelsCustom.cc"], **extension_kwargs),
                   Extension('_PostsynapticModels', ["pygenn/genn_wrapper/generated/PostsynapticModels.i", "pygenn/genn_wrapper/generated/newPostsynapticModelsCustom.cc"], **extension_kwargs),
                   Extension('_WeightUpdateModels', ["pygenn/genn_wrapper/generated/WeightUpdateModels.i", "pygenn/genn_wrapper/generated/newWeightUpdateModelsCustom.cc"], **extension_kwargs),
                   Extension('_CurrentSourceModels', ["pygenn/genn_wrapper/generated/CurrentSourceModels.i", "pygenn/genn_wrapper/generated/currentSourceModelsCustom.cc"], **extension_kwargs)])
