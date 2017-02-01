#!/bin/bash
# By default no flags are passed to genn-buildmodel.sh
BUILD_FLAGS=""

# Parse command line arguments
OPTIND=1
while getopts "c" opt; do
    case "$opt" in
    c)  BUILD_FLAGS="-c"
        ;;
    esac
done

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Loop through feature tests
for f in features/*;
    do
        echo "Calculating code generation coverage of $f..."

        # Push feature directory
        pushd $f

        # Clean
        make clean &>msg

        # Build and generate model (measuring coverage)
        genn-buildmodel.sh $BUILD_FLAGS -v model.cc &>msg || exit $?
	
        # Pop feature directory
        popd
    done;
