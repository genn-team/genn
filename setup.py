#!/usr/bin/env python
import numpy as np
import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

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

pygenn_path = os.path.join(genn_path, "pygenn")
pygenn_include = os.path.join(pygenn_path, "include")
pygenn_swig = os.path.join(pygenn_path, "swig")
pygenn_generated = os.path.join(pygenn_path, "generated")
genn_include = os.path.join(genn_path, "lib", "include")

swig_opts = ["-c++", "-outdir", pygenn_path, "-I" + pygenn_include, "-I" + pygenn_generated, 
             "-I" + pygenn_swig, "-I" + genn_include]

include_dirs = [genn_include, pygenn_include, pygenn_generated,
                os.path.join(cuda_path, "include"),
                os.path.join(numpy_path, "core", "include")]

libraries =["cuda", "cudart", "genn_DYNAMIC"]

library_dirs = [os.path.join(cuda_path, "lib64"), pygenn_path]

genn_wrapper = Extension('_genn_wrapper', [
    "pygenn/generated/genn_wrapper.i",
    "lib/src/generateALL.cc", "lib/src/generateCPU.cc", 
    "lib/src/generateInit.cc", "lib/src/generateKernels.cc", 
    "lib/src/generateMPI.cc", "lib/src/generateRunner.cc",
    "pygenn/generated/currentSourceModelsCustom.cc",
    "pygenn/generated/initVarSnippetCustom.cc",
    "pygenn/generated/newNeuronModelsCustom.cc",
    "pygenn/generated/newPostsynapticModelsCustom.cc",
    "pygenn/generated/newWeightUpdateModelsCustom.cc"], 
    swig_opts=swig_opts, include_dirs=include_dirs, 
    define_macros=[("GENERATOR_MAIN_HANDLED", None), ("NVCC", "\"" + os.path.join(cuda_path, "bin", "nvcc") + "\"")],
    libraries=libraries, library_dirs=library_dirs)

setup(name = "pygenn",
      version = "0.1",
      packages = find_packages(),
      url="https://github.com/genn-team/genn",
      author="University of Sussex",
      description="Python interface to the GeNN simulator",
      ext_package="pygenn",
      ext_modules=[genn_wrapper,
                   Extension('_Snippet', ["pygenn/swig/Snippet.i"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_NewModels', ["pygenn/swig/NewModels.i"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_GeNNPreferences', ["pygenn/swig/GeNNPreferences.i"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_StlContainers', ["pygenn/generated/StlContainers.i"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_SharedLibraryModel', ["pygenn/generated/SharedLibraryModel.i"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_InitVarSnippet', ["pygenn/generated/InitVarSnippet.i", "pygenn/generated/initVarSnippetCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_NeuronModels', ["pygenn/generated/NeuronModels.i", "pygenn/generated/newNeuronModelsCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_PostsynapticModels', ["pygenn/generated/PostsynapticModels.i", "pygenn/generated/newPostsynapticModelsCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_WeightUpdateModels', ["pygenn/generated/WeightUpdateModels.i", "pygenn/generated/newWeightUpdateModelsCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs),
                   Extension('_CurrentSourceModels', ["pygenn/generated/CurrentSourceModels.i", "pygenn/generated/currentSourceModelsCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs)])
