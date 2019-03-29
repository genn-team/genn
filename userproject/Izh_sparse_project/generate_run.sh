#!/bin/bash

generate_runner_help () {
    echo "generate_runner.sh script usage:"
    echo "generate_runner.sh [dfc] <nNeurons> <nConn> <gscale> <input fac> <outdir>"
    
    # Call function 
    generate_runner_flags_help
}

# Run command line parser
source ../include/parse_command_line.sh

# Check remaining parameter count
if [[ $# != 5 ]]; then generate_runner_help; exit; fi

# Read remaing parameters
NNEURONS="$1"
NCONN="$2"
GSCALE="$3"
INPUT_FAC="$4"
OUTDIR="$5"

# Write sizes header
echo "#pragma once" > sizes.h
echo "#define _NNeurons $NNEURONS" >> sizes.h
echo "#define _NConn $NCONN" >> sizes.h
echo "#define _GScale $GSCALE" >> sizes.h
echo "#define _InputFac $INPUT_FAC" >> sizes.h
echo "#define _FTYPE GENN_$FTYPE" >> sizes.h
  
# Generate model code
genn-buildmodel.sh $BUILD_MODEL_ARGS IzhSparse.cc

# Build model code
make

# Run
if [[ -n "$DEBUG" ]]; then
    gdb -tui --args IzhSparse.cc "$OUTDIR"
else
    ./PoissonIzh "$OUTDIR"
fi
