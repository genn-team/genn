import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext, WIN
from setuptools import find_packages, setup

# Determine correct suffix for GeNN libraries
if WIN:
    genn_lib_suffix = "_Release_DLL"# "_Debug_DLL" if debug_build else "_Release_DLL"
else:
    genn_lib_suffix = "_dynamic"#"_dynamic_debug" if debug_build else "_dynamic"

genn_path = os.path.dirname(os.path.abspath(__file__))

pygenn_path = os.path.join(genn_path, "pygenn")
genn_wrapper_path = os.path.join(pygenn_path, "genn_wrapper")
genn_wrapper_src = os.path.join(genn_wrapper_path, "src")
genn_include = os.path.join(genn_path, "include", "genn", "genn")
genn_third_party_include = os.path.join(genn_path, "include", "genn", "third_party")

ext_modules = [
    Pybind11Extension("genn",
        [os.path.join(genn_wrapper_src, "genn.cc")],
        include_dirs=[genn_include, genn_third_party_include],
        library_dirs=[genn_wrapper_path],
        libraries=["genn" + genn_lib_suffix],
        cxx_std=17)]
    
    
setup(
    name="pygenn2",
    version="0.0.1",
    packages = find_packages(),
    url="https://github.com/genn_team/genn",
    ext_package="pygenn.genn_wrapper",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
)
