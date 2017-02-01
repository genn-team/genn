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

# Loop through feature tests
NUM_SUCCESSES=0
NUM_FAILURES=0
for f in features/*;
    do
        echo "Running test $f..."

        # Push feature directory
        pushd $f

        # Clean
        make clean &>msg

        # Build and generate model
        genn-buildmodel.sh $BUILD_FLAGS model.cc &>msg || exit $?
	
        # Build
        make $MAKE_FLAGS &>msg || exit $?

        # Run tests
        ./test --gtest_output="xml:test_results.xml"
        if [ $? -eq 0 ]; then
            NUM_SUCCESSES=$((NUM_SUCCESSES+1))
        else
            NUM_FAILURES=$((NUM_FAILURES+1))
        fi

        # Pop feature directory
        popd
    done;

# Print brief summary of output
echo "$NUM_SUCCESSES feature tests succeeded"
echo "$NUM_FAILURES feature tests failed"
