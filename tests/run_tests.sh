#!/bin/bash


# Loop through feature tests
# **TODO** CPU_ONLY ness should be passed as command line argument based on node configuration
for f in features/*;
    do
        # Push feature directory
        pushd $f

        # Clean
        make clean

        # Build and generate model,
        genn-buildmodel.sh -c -v model.cc

        # Build
        make CPU_ONLY=1

        # Run tests
        # **NOTE** we're assuming that optimisation level doesn't effect code generation
        ./test --gtest_output="xml:test_results.xml"

        # Pop feature directory
        popd
    done;


#