#!/bin/bash
BUILD_FLAGS =  ""
MAKE_FLAGS = ""

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

NUM_SUCCESSES=0
NUM_FAILURES=0

# Loop through feature tests
# **TODO** CPU_ONLY ness should be passed as command line argument based on node configuration
for f in features/*;
    do
        # Push feature directory
        pushd $f

        # Clean
        make clean

        # Build and generate model
        genn-buildmodel.sh $BUILD_FLAGS model.cc || exit $?
	
        # Build
        make $MAKE_FLAGS || exit $?

        # Run tests
        # **NOTE** we're assuming that optimisation level doesn't effect code generation
        ./test --gtest_output="xml:test_results.xml"
        if [ $? -eq 0 ]; then
            NUM_SUCCESSES=$((NUM_SUCCESSES+1))
        else
            NUM_FAILURES=$((NUM_FAILURES+1))
        fi
        # Pop feature directory
        popd
    done;

echo "$NUM_SUCCESSES feature tests succeeded"
echo "$NUM_FAILURES feature tests failed"
