#!/bin/bash
BUILD_FLAGS =  ""
MAKE_FLAGS = ""

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Loop through feature tests
# **TODO** CPU_ONLY ness should be passed as command line argument based on node configuration
for f in features/*;
    do
        # Push feature directory
        pushd $f

        # Clean
        make clean

        # Build and generate model (measuring coverage)
        genn-buildmodel.sh $BUILD_FLAGS -v model.cc || exit $?
	
        # Pop feature directory
        popd
    done;
