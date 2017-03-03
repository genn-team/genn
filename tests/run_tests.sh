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

# Zero counters of test passes and fails
NUM_SUCCESSES=0
NUM_FAILURES=0

# Loop through feature tests
for f in features/*;
    do
        echo "Running test $f..."

        # Push feature directory
        pushd $f
        
        # Loop through model suffixes
        for s in "" _new;
            do
                # Clean
                make clean 1>> ../../msg 2>> ../../msg

                # Build and generate model
                genn-buildmodel.sh $BUILD_FLAGS model$s.cc 1>>../../msg 2>> ../../msg || exit $?
                
                # Build
                make $MAKE_FLAGS 1>>../../msg 2>>../../msg || exit $?

                # Run tests
                ./test --gtest_output="xml:test_results$s.xml"
                if [ $? -eq 0 ]; then
                    NUM_SUCCESSES=$((NUM_SUCCESSES+1))
                else
                    NUM_FAILURES=$((NUM_FAILURES+1))
                fi
            done;

        # Pop feature directory
        popd
    done;

# Enter unit tests directory
pushd unit

# Clean
make clean 1>> ../../msg 2>> ../../msg

# Build
make $MAKE_FLAGS 1>>../../msg 2>>../../msg || exit $?

# Run tests
./test --gtest_output="xml:test_results$s.xml"
if [ $? -eq 0 ]; then
    NUM_SUCCESSES=$((NUM_SUCCESSES+1))
else
    NUM_FAILURES=$((NUM_FAILURES+1))
fi

# Pop unit tests directory
popd

# Print brief summary of output
echo "$NUM_SUCCESSES tests succeeded"
echo "$NUM_FAILURES tests failed"
