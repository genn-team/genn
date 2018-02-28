#!/bin/bash

# By default no flags are passed to genn-buildmodel.sh or make
BUILD_FLAGS=""
MAKE_FLAGS=""

# Parse command line arguments
OPTIND=1
while getopts "c" opt; do
    case "$opt" in
    c)  BUILD_FLAGS="-c"
        MAKE_FLAGS="CPU_ONLY=1"
        ;;
    esac
done

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Delete existing output
rm -f msg

# Loop through feature tests
for f in features/*;
    do
        echo "Running test $f..."

        # Push feature directory
        pushd $f
        
        # Loop through model suffixes
        for s in "" _new;
            do
                if [ -f "model$s.cc" ]; then
                    # Determine where the sim code is located for this test
                    c=$(basename $f)$s"_CODE"

                    # Clean
                    make $MAKE_FLAGS SIM_CODE=$c clean 1>> ../../msg 2>> ../../msg

                    # Build and generate model
                    if genn-buildmodel.sh $BUILD_FLAGS model$s.cc 1>>../../msg 2>> ../../msg ; then
                        # Make
                        if make $MAKE_FLAGS SIM_CODE=$c 1>>../../msg 2>>../../msg ; then
                            # Run tests
                            ./test --gtest_output="xml:test_results$s.xml"
                        fi
                    fi
                fi
            done;

        # Pop feature directory
        popd
    done;


# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Run unit tests
pushd unit

# Clean
make clean 1>> ../msg 2>> ../msg

# Build
make $MAKE_FLAGS 1>>../msg 2>>../msg 

# Run tests
./test --gtest_output="xml:test_results_unit.xml"

# Pop unit tests directory
popd

# Run SpineML tests
pushd spineml
pushd simulator

# Clean
make clean 1>> ../../msg 2>> ../../msg

# Build
make $MAKE_FLAGS 1>>../../msg 2>>../../msg 

# Run SpineML simulator tests
./test --gtest_output="xml:test_results_spineml.xml"

popd    # simulator
popd    # spineml

