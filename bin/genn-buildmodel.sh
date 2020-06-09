#!/bin/bash

# display genn-buildmodel.sh help
genn_help () {
    echo "genn-buildmodel.sh script usage:"
    echo "genn-buildmodel.sh [cdho] model"
    echo "-c            only generate simulation code for the CPU"
    echo "-l            generate simulation code for OpenCL"
    echo "-d            enables the debugging mode"
    echo "-m            generate MPI simulation code"
    echo "-v            generates coverage information"
    echo "-h            shows this help message"
    echo "-s standard   changes the C++ standard the model is built with"
    echo "-o outpath    changes the output directory"
    echo "-i includepath    add additional include directories (seperated by colons)"
}

# handle script errors
genn_error () { # $1=line, $2=code, $3=message
    echo "genn-buildmodel.sh:$1: error $2: $3"
    exit "$2"
}
trap 'genn_error $LINENO 50 "command failure"' ERR

# parse command options
OUT_PATH="$PWD";
BUILD_MODEL_INCLUDE=""
GENERATOR_MAKEFILE="MakefileCUDA"
CXX_STANDARD="c++11"
while [[ -n "${!OPTIND}" ]]; do
    while getopts "cldmvs:o:i:h" option; do
    case $option in
        c) GENERATOR_MAKEFILE="MakefileSingleThreadedCPU";;
        l) GENERATOR_MAKEFILE="MakefileOpenCL";;
        d) DEBUG=1;;
        m) MPI_ENABLE=1;;
        v) COVERAGE=1;;
        h) genn_help; exit;;
        s) CXX_STANDARD="$OPTARG";;
        o) OUT_PATH="$OPTARG";;
        i) BUILD_MODEL_INCLUDE="$OPTARG";;
        ?) genn_help; exit;;
    esac
    done
    if [[ $OPTIND > $# ]]; then break; fi
    MODEL="${!OPTIND}"
    let OPTIND++
done

if [[ -z "$MODEL" ]]; then
    genn_error $LINENO 2 "no model file given"
fi
pushd $OUT_PATH > /dev/null
OUT_PATH="$PWD"
popd > /dev/null
pushd $(dirname $MODEL) > /dev/null
MACROS="MODEL=$PWD/$(basename $MODEL) GENERATOR_PATH=$OUT_PATH BUILD_MODEL_INCLUDE=$BUILD_MODEL_INCLUDE CXX_STANDARD=$CXX_STANDARD"
GENERATOR=./generator
popd > /dev/null
if [[ -n "$DEBUG" ]]; then
    MACROS="$MACROS DEBUG=1";
    GENERATOR="$GENERATOR"_debug
fi

if [[ -n "$MPI_ENABLE" ]]; then
    MACROS="$MACROS MPI_ENABLE=1";
    GENERATOR="$GENERATOR"_mpi
fi

if [[ -n "$COVERAGE" ]]; then
    MACROS="$MACROS COVERAGE=1";
    GENERATOR="$GENERATOR"_coverage
fi

# If CUDA path isn't set, default to standard path for (at least Ubuntu) Linux systems
# **NOTE** setting CUDA_PATH is a REQUIRED post-installation action when installing CUDA so this shouldn't be required
export CUDA_PATH=${CUDA_PATH-/usr/local/cuda} 

# Count cores using approach lifted from https://stackoverflow.com/questions/6481005/how-to-obtain-the-number-of-cpus-cores-in-linux-from-the-command-line
if [[ $(uname) = "Darwin" ]]; then
    CORE_COUNT=$(sysctl -n hw.physicalcpu_max)
else
    CORE_COUNT=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

# generate model code
BASEDIR=$(dirname "$0")
make -j $CORE_COUNT -C $BASEDIR/../src/genn/generator -f $GENERATOR_MAKEFILE $MACROS

if [[ -n "$MPI_ENABLE" ]]; then
    cp "$GENERATOR" "$GENERATOR"_"$OMPI_COMM_WORLD_RANK"
fi

if [[ -n "$DEBUG" ]]; then
    gdb -tui --args "$GENERATOR" "$OUT_PATH"
else
    if [[ -n "$MPI_ENABLE" ]]; then
        "$GENERATOR"_"$OMPI_COMM_WORLD_RANK" "$OUT_PATH"
    else
        "$GENERATOR" "$OUT_PATH"
    fi
fi

echo "model build complete"
