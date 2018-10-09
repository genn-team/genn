#!/usr/bin/env python
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
genn_dir = os.path.dirname(os.path.abspath(__file__))

swig_opts = ["-c++", 
             "-I" + os.path.join(genn_dir, "pygenn"), 
             "-I" + os.path.join(genn_dir, "lib")]

include_dirs = [os.path.join(genn_dir, "lib", "include"),
                os.path.join(genn_dir, "pygenn", "include"),
                os.path.join(os.environ["CUDA_PATH"], "include")]

setup(name = "pygenn",
      version = "0.1",
      packages = find_packages(),
      url="https://github.com/project-rig/pynn_spinnaker",
      author="University of Sussex",
      description="Python interface to the GeNN simulator",
      ext_modules=[Extension('pygenn', ["pygenn/generated/pygenn.i"], swig_opts=swig_opts, include_dirs=include_dirs),
                   Extension('pygenn.NeuronModels', ["pygenn/generated/NeuronModels.i", "pygenn/generated/newNeuronModelsCustom.cc"], swig_opts=swig_opts, include_dirs=include_dirs)])
