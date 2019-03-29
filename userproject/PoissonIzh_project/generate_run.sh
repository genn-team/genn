#!/bin/bash

generate_runner_help () {
    echo "generate_runner.sh script usage:"
    echo "generate_runner.sh [dfc] <nPoisson> <nIzh> <pConn> <gscale> <outdir>"
    
    # Call function 
    generate_runner_flags_help
}

# Run command line parser
source ../include/parse_command_line.sh

# Check remaining parameter count
if [[ $# != 5 ]]; then generate_runner_help; exit; fi

# Read remaing parameters
NPOISSON="$1"
NIZH="$2"
PCONN="$3"
GSCALE="$4"
OUTDIR="$5"

# Write sizes header
echo "#pragma once" > sizes.h
echo "#define _NPoisson $NPOISSON" >> sizes.h
echo "#define _NIzh $NIZH" >> sizes.h
echo "#define _PConn $PCONN" >> sizes.h
echo "#define _GScale $GSCALE" >> sizes.h
echo "#define _FTYPE GENN_$FTYPE" >> sizes.h
  
# Generate model code
genn-buildmodel.sh $BUILD_MODEL_ARGS PoissonIzh.cc

# Build model code
make

# Run
if [[ -n "$DEBUG" ]]; then
    gdb -tui --args PoissonIzh "$OUTDIR"
else
    ./PoissonIzh "$OUTDIR"
fi
