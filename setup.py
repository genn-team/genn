import os
import sys
from copy import deepcopy
from platform import system, uname
from psutil import cpu_count
from subprocess import check_call
from pybind11.setup_helpers import Pybind11Extension, build_ext, WIN, MACOS
from setuptools import find_packages, setup

# Loop through command line arguments
debug_build = False
coverage_build = False
build_genn_libs = True
filtered_args = []
for arg in sys.argv:
    if arg == "--debug":
        debug_build = True
    elif arg == "--coverage":
        coverage_build = True
        continue
    elif arg in ["clean", "egg_info", "sdist"]:
        build_genn_libs = False

    filtered_args.append(arg)

# Add filtered (those that setuptools will understand) back to sys.argv
sys.argv = filtered_args

# Get CUDA path from environment variable - setting this up is a required CUDA post-install step
cuda_path = os.environ.get("CUDA_PATH")

# Is CUDA installed?
cuda_installed = cuda_path is not None and os.path.exists(cuda_path)

# Get HIP path from environment variable
hip_path = os.environ.get("HIP_PATH")

# Is HIP installed
hip_installed = hip_path is not None and os.path.exists(hip_path)

# Are we on Linux?
# **NOTE** Pybind11Extension provides WIN and MAC
LINUX = system() == "Linux"

# Are we on WSL?
if sys.version_info < (3, 3):
    WSL = "microsoft" in uname()[2]
else:
    WSL = "microsoft" in uname().release

# Determine correct suffix for GeNN libraries
if WIN:
    genn_lib_suffix = "_Debug_DLL" if debug_build else "_Release_DLL"
else:
    if coverage_build:
        genn_lib_suffix = "_coverage_dynamic"
    elif debug_build:
        genn_lib_suffix = "_dynamic_debug"
    else:
        genn_lib_suffix = "_dynamic"

abs_genn_path = os.path.dirname(os.path.abspath(__file__))
genn_include = os.path.join(".", "include", "genn", "genn")
genn_third_party_include = os.path.join(".", "include", "genn", "third_party")
pygenn_path = os.path.join(".", "pygenn")
pygenn_src = os.path.join(pygenn_path, "src")
pygenn_include = os.path.join(pygenn_path, "include")

# Always package LibGeNN
if WIN:
    package_data = ["genn" + genn_lib_suffix + ".dll",
                    "libffi" + genn_lib_suffix + ".dll"]
elif MACOS:
    package_data = ["libgenn" + genn_lib_suffix + ".dylib"]
else:
    package_data = ["libgenn" + genn_lib_suffix + ".so"]

# Define standard kwargs for building all extensions
genn_extension_kwargs = {
    "include_dirs": [pygenn_include, genn_include, genn_third_party_include],
    "library_dirs": [pygenn_path],
    "libraries": ["genn" + genn_lib_suffix],
    "cxx_std": 17,
    "extra_compile_args": [],
    "extra_link_args": [],
    "define_macros": [("LINKING_GENN_DLL", 1), ("LINKING_BACKEND_DLL", 1)]}

# If this is Windows
if WIN:
    # Turn off warnings about dll-interface being required for stuff to be
    # used by clients and prevent windows.h exporting TOO many awful macros
    genn_extension_kwargs["extra_compile_args"].extend(["/wd4251", "-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR"])

    # Add include directory for FFI as it's built from source
    genn_extension_kwargs["include_dirs"].append(os.path.join(genn_third_party_include, "libffi"))

    # Add FFI library with correct suffix
    # **TODO** just call this ffi
    genn_extension_kwargs["libraries"].append("libffi" + genn_lib_suffix)

    # Add GeNN and FFI libraries to dependencies
    genn_extension_kwargs["depends"] = [os.path.join(pygenn_path, "genn" + genn_lib_suffix + ".dll"),
                                        os.path.join(pygenn_path, "libffi" + genn_lib_suffix + ".dll")]
# Otherwise
else:
    # --- Linux/macOS libffi linkage ---
    genn_extension_kwargs["libraries"].append("ffi")
    # Add GeNN library to dependencies
    if MACOS:
        genn_extension_kwargs["depends"] = [
            os.path.join(pygenn_path, "libgenn" + genn_lib_suffix + ".dylib"),
            os.path.join(pygenn_src, "docStrings.h")]
        # macOS: ensure extensions can find bundled dylibs next to them
        genn_extension_kwargs["extra_link_args"].extend(["-Wl,-rpath,@loader_path"])
    else:
        genn_extension_kwargs["depends"] = [
            os.path.join(pygenn_path, "libgenn" + genn_lib_suffix + ".so"),
            os.path.join(pygenn_src, "docStrings.h")]
    # If this is Linux, we want to add extension directory i.e. $ORIGIN to runtime
    # directories so libGeNN and backends can be found wherever package is installed
    if LINUX:
        genn_extension_kwargs["runtime_library_dirs"] = ["$ORIGIN"]      

if coverage_build:
    if LINUX:
        genn_extension_kwargs["extra_compile_args"].append("--coverage")
        genn_extension_kwargs["extra_link_args"].append("--coverage")
    elif MACOS:
        genn_extension_kwargs["extra_compile_args"].extend(["-fprofile-instr-generate", "-fcoverage-mapping"])

# By default build single-threaded CPU backend
backends = [("single_threaded_cpu", "singleThreadedCPU", {})]

# If CUDA was found, add backend configuration
if cuda_installed:
    # Get CUDA library directory
    # **NOTE** $(CUDA_PATH)/lib/x64 is correct for system CUDA installs on Windows but not conda
    cuda_library_dirs = []
    if MACOS:
        cuda_library_dirs.append(os.path.join(cuda_path, "lib"))
    elif WIN:
        cuda_library_dirs.append(os.environ.get("CUDA_LIBRARY_PATH", os.path.join(cuda_path, "lib", "x64")))
    else:
        cuda_library_dirs.append(os.path.join(cuda_path, "lib64"))

    # If we're running on WSL, add additional library path so libcuda can be found
    if WSL:
        cuda_library_dirs.append("/usr/lib/wsl/lib")

    # Add backend
    # **NOTE** on Mac OS X, a)runtime_library_dirs doesn't work b)setting rpath is required to find CUDA
    backends.append(("cuda", "cuda",
                     {"libraries": ["cuda", "cudart"],
                      "include_dirs": [os.path.join(cuda_path, "include")],
                      "library_dirs": cuda_library_dirs,
                      "extra_link_args": ["-Wl,-rpath," + cuda_library_dirs[0]] if MACOS else []}))

# If HIP was found, add backend configuration
if hip_installed:
    # If we're using NVIDIA
    hip_include_dirs = [os.path.join(hip_path, "include")]
    hip_library_dirs = [os.path.join(hip_path, "lib")]
    hip_libraries = ["hiprand"]
    hip_extra_compile_args = []
    if os.environ.get("HIP_PLATFORM") == "nvidia":
        assert cuda_installed
        # Add CUDA include directories
        hip_include_dirs.append(os.path.join(cuda_path, "include"))
            
        # Add CUDA library directories
        if WIN:
            hip_library_dirs.append(os.path.join(cuda_path, "lib", "x64"))
        else:
            hip_library_dirs.append(os.path.join(cuda_path, "lib64"))
        
        # If we're running on WSL, add additional library path so libcuda can be found
        if WSL:
            hip_library_dirs.append("/usr/lib/wsl/lib")
        
        # Add CUDA libraries
        hip_libraries.extend(["cuda", "cudart"])
        
        # Set platform compile arguments
        hip_extra_compile_args.append("-D__HIP_PLATFORM_NVIDIA__")
    else:
        hip_libraries.extend(["amdhip64"])
        hip_extra_compile_args.append("-D__HIP_PLATFORM_AMD__")      
    
    # Add backend
    # **NOTE** on Mac OS X, a)runtime_library_dirs doesn't work b)setting rpath is required to find CUDA
    backends.append(("hip", "hip",
                     {"libraries": hip_libraries,
                      "include_dirs": hip_include_dirs,
                      "library_dirs": hip_library_dirs,
                      "extra_compile_args": hip_extra_compile_args}))

ext_modules = [
    Pybind11Extension("_runtime",
                      [os.path.join(pygenn_src, "runtime.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("_genn",
                      [os.path.join(pygenn_src, "genn.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("types",
                      [os.path.join(pygenn_src, "types.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("init_sparse_connectivity_snippets",
                      [os.path.join(pygenn_src, "initSparseConnectivitySnippets.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("init_toeplitz_connectivity_snippets",
                      [os.path.join(pygenn_src, "initToeplitzConnectivitySnippets.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("init_var_snippets",
                      [os.path.join(pygenn_src, "initVarSnippets.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("current_source_models",
                      [os.path.join(pygenn_src, "currentSourceModels.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("custom_connectivity_update_models",
                      [os.path.join(pygenn_src, "customConnectivityUpdateModels.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("custom_update_models",
                      [os.path.join(pygenn_src, "customUpdateModels.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("neuron_models",
                      [os.path.join(pygenn_src, "neuronModels.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("postsynaptic_models",
                      [os.path.join(pygenn_src, "postsynapticModels.cc")],
                      **genn_extension_kwargs),
    Pybind11Extension("weight_update_models",
                      [os.path.join(pygenn_src, "weightUpdateModels.cc")],
                      **genn_extension_kwargs)]

# Loop through namespaces of supported backends
for module_stem, source_stem, kwargs in backends:
    # Take a copy of the standard extension kwargs
    backend_extension_kwargs = deepcopy(genn_extension_kwargs)

    # Extend any settings specified by backend
    for n, v in kwargs.items():
        backend_extension_kwargs[n].extend(v)

    # Add relocatable version of backend library to libraries
    # **NOTE** this is added BEFORE libGeNN as this library needs symbols FROM libGeNN
    if WIN:
        backend_extension_kwargs["depends"].append(
            os.path.join(pygenn_path, "genn_" + module_stem + "_backend" + genn_lib_suffix + ".dll"))

        package_data.append("genn_" + module_stem + "_backend" + genn_lib_suffix + ".dll")
    elif MACOS:
        backend_extension_kwargs["depends"].append(
            os.path.join(pygenn_path, "libgenn_" + module_stem + "_backend" + genn_lib_suffix + ".dylib"))
        package_data.append("libgenn_" + module_stem + "_backend" + genn_lib_suffix + ".dylib")    
    else:
        backend_extension_kwargs["depends"].append(
            os.path.join(pygenn_path, "libgenn_" + module_stem + "_backend" + genn_lib_suffix + ".so"))

        package_data.append("libgenn_" + module_stem + "_backend" + genn_lib_suffix + ".so")

    # Add backend include directory to both SWIG and C++ compiler options
    backend_include_dir = os.path.join(".", "include", "genn", "backends", module_stem)
    backend_extension_kwargs["libraries"].insert(0, "genn_" + module_stem + "_backend" + genn_lib_suffix)
    backend_extension_kwargs["include_dirs"].append(backend_include_dir)

    # Add extension to list
    ext_modules.append(Pybind11Extension(module_stem + "_backend", 
                                         [os.path.join(pygenn_src, source_stem + "Backend.cc")],
                                         **backend_extension_kwargs))

    # If we should build required GeNN libraries
    if build_genn_libs:
        # If compiler is MSVC
        if WIN:
            # **NOTE** ensure pygenn_path has trailing slash to make MSVC happy
            out_dir = os.path.join(abs_genn_path, "pygenn", "")
            check_call(["msbuild", "genn.sln", f"/t:{module_stem}_backend",
                        f"/p:Configuration={genn_lib_suffix[1:]}",
                        "/m", "/verbosity:quiet",
                        f"/p:OutDir={out_dir}"],
                        cwd=abs_genn_path)
        else:
            # Define make arguments
            make_arguments = ["make", f"{module_stem}_backend", "DYNAMIC=1",
                              f"LIBRARY_DIRECTORY={os.path.join(abs_genn_path, 'pygenn')}",
                              f"--jobs={cpu_count(logical=False)}"]
            if debug_build:
                make_arguments.append("DEBUG=1")

            if coverage_build:
                make_arguments.append("COVERAGE=1")

            # Build
            check_call(make_arguments, cwd=abs_genn_path)

# Read version from txt file
with open(os.path.join(abs_genn_path, "version.txt")) as version_file:
    version = version_file.read().strip()

setup(
    name="pygenn",
    version=version,
    packages = find_packages(),
    package_data={"pygenn": package_data},

    url="https://github.com/genn-team/genn",
    ext_package="pygenn",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy>=1.17", "psutil",
                      "importlib-metadata>=1.0;python_version<'3.8'",
                      "setuptools"],
    extras_require={
        "doc": ["sphinx", "sphinx-gallery", "sphinx-argparse"],
        "userproject": ["mnist", "tqdm", "scipy", "matplotlib"],
        "test": ["bitarray", "pytest", "flaky", "pytest-cov", "cupy-cuda12x"]})
