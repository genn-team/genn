#!/bin/bash

# display genn-buildmodel.sh help
genn_help () {
    echo "=== genn-buildmodel.sh script usage ==="
    echo "genn-buildmodel.sh [cdho] model"
    echo "-c            only generate simulation code for the CPU"
    echo "-d            enables the debugging mode"
    echo "-h            shows this help message"
    echo "-o outpath    changes the output directory"
}

# handle script errors
genn_error () { # $1=line, $2=code, $3=message
    echo "genn-buildmodel.sh:$1: error $2: $3"
    exit "$2"
}
trap 'genn_error $LINENO 50 "command failure"' ERR

# parse command options
OUT_PATH="$PWD";
while [[ -n "${!OPTIND}" ]]; do
    while getopts "cdo:h" option; do
    case $option in
        c) CPU_ONLY=1;;
        d) DEBUG=1;;
        h) genn_help; exit;;
        o) OUT_PATH="$OPTARG";;
        ?) genn_help; exit;;
    esac
    done
    if [[ $OPTIND > $# ]]; then break; fi
    MODEL="${!OPTIND}"
    let OPTIND++
done

# command options logic
if [[ -z "$GENN_PATH" ]]; then
    if [[ $(uname -s) == "Linux" ]]; then
        echo "GENN_PATH is not defined - trying to auto-detect"
        export GENN_PATH="$(readlink -f $(dirname $0)/../..)"
    else
        genn_error $LINENO 1 "GENN_PATH is not defined"
    fi
fi
if [[ -z "$MODEL" ]]; then
    genn_error $LINENO 2 "no model file given"
fi
pushd $OUT_PATH > /dev/null
OUT_PATH="$PWD"
popd > /dev/null
pushd $(dirname $MODEL) > /dev/null
MACROS="MODEL=$PWD/$(basename $MODEL)"
popd > /dev/null
if [[ -n "$DEBUG" ]]; then MACROS="$MACROS DEBUG=1"; fi
if [[ -n "$CPU_ONLY" ]]; then MACROS="$MACROS CPU_ONLY=1"; fi

# generate model code
cd "$OUT_PATH"
make clean -f "$GENN_PATH/lib/src/GNUmakefile"
if [[ -n "$DEBUG" ]]; then
    echo "debugging mode ON"
    make debug -f "$GENN_PATH/lib/src/GNUmakefile" $MACROS
    gdb -tui --args ./generateALL "$OUT_PATH"
else
    make -f "$GENN_PATH/lib/src/GNUmakefile" $MACROS
    ./generateALL "$OUT_PATH"
fi

echo "model build complete"
