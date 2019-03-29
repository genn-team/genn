#!/bin/bash

generate_runner_help () {
    echo "generate_runner.sh script usage:"
    echo "generate_runner.sh [dfc] <nC1> <outdir>"
    
    # Call function 
    generate_runner_flags_help
}

# Run command line parser
source ../include/parse_command_line.sh

# Check remaining parameter count
if [[ $# != 2 ]]; then generate_runner_help; exit; fi

# Read remaing parameters
NC1="$1"
OUTDIR="$2"

# Write sizes header
echo "#pragma once" > sizes.h
echo "#define _NC1 $NC1" >> sizes.h
echo "#define _FTYPE GENN_$FTYPE" >> sizes.h

echo "$#"
if [[ -n "$DEBUG" ]]; then
    echo "DEBUG!"
fi

# Generate model code
genn-buildmodel.sh $BUILD_MODEL_ARGS OneComp.cc

# Build model code
make

if [[ -n "$DEBUG" ]]; then
    gdb -tui --args "$GENERATOR" "$OUT_PATH"
else
    if [[ -n "$MPI_ENABLE" ]]; then
        "$GENERATOR"_"$OMPI_COMM_WORLD_RANK" "$OUT_PATH"
    else
        "$GENERATOR" "$OUT_PATH"
    fi
fi
