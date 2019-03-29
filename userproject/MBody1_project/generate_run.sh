#!/bin/bash

generate_runner_help () {
    echo "generate_runner.sh script usage:"
    echo "generate_runner.sh [dfcby] <nAL> <nKC> <nLHI> <nLb> <gscale>  <outdir>"
    echo "-d            run debugger"
    echo "-f ftype      what floating point type to use (FLOAT or DOUBLE)"
    echo "-c            use CPU only backend"
    echo "-b            use bitmask (rather than dense) data structure for PN->KC connectivity"
    echo "-t            whether to simulate delays of (5 * DT) ms on KC->DN and of (3 * DT) ms on DN->DN synapse population"

}

# Parse flags
GENERATOR_MAKEFILE="MakefileCUDA"
FTYPE="FLOAT"
while getopts ":df:cbyh" option; do
    case $option in
        d) BUILD_MODEL_ARGS="$BUILD_MODEL_ARGS -d";DEBUG=1;;
        f) FTYPE="$OPTARG";;
        c) BUILD_MODEL_ARGS="$BUILD_MODEL_ARGS -c";;
        b) BITMASK=1;;
        y) DELAYED=1;;
        h) generate_runner_help; exit;;
        ?) generate_runner_help; exit;;
        :) generate_runner_help; exit;;
    esac
done

# Shift off arguments already processed
shift $((OPTIND -1))

# Check remaining parameter count
if [[ $# != 6 ]]; then generate_runner_help; exit; fi


# Read remaing parameters
NAL="$1"
NKC="$2"
NLHI="$3"
NLB="$4"
GSCALE="$5"
OUTDIR="$6"

# Write sizes header
echo "#pragma once" > sizes.h
echo "#define _NAL $NAL" >> sizes.h
echo "#define _NKC $NKC" >> sizes.h
echo "#define _NLHI $NLHI" >> sizes.h
echo "#define _NLB $NLB" >> sizes.h
echo "#define _GScale $GSCALE" >> sizes.h
echo "#define _FTYPE GENN_$FTYPE" >> sizes.h
if [[ -n "$BITMASK" ]]; then echo "#define BITMASK" >> sizes.h; fi
if [[ -n "$DELAYED" ]]; then echo "#define DELAYED_SYNAPSES" >> sizes.h; fi

# Generate model code
genn-buildmodel.sh $BUILD_MODEL_ARGS MBody1.cc

# Build model code
make

# Run
if [[ -n "$DEBUG" ]]; then
    gdb -tui --args IzhSparse.cc "$OUTDIR"
else
    ./PoissonIzh "$OUTDIR"
fi
