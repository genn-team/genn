#!/bin/bash

# reset_coverage () {
#     # On OSX remove existing raw coverage files before running each test
#     # **NOTE** GCC can successfully combine gcno and gcda files itself but not LLVM
#     # **NOTE** only remove libgenn coverage COUNTS as libgenn itself and hence its .gcdo files don't get rebuilt
#     if [[ "$(uname)" = "Darwin" ]]; then
#         rm -f *.gcno *.gcda
#         rm -rf $GENN_PATH/lib/**/*.gcda
#     fi
# }
# 
# update_coverage () {
#     # On OSX convert the coverage of each test to LCOV format
#     if [[ "$(uname)" = "Darwin" ]]; then
#         # Capture GCOV output for this test
#         lcov --directory $GENN_PATH --capture -rc lcov_branch_coverage=1 --output-file $1.txt 1>> ../../msg 2>> ../../msg
# 
#         # Add this test's output to LCOV command line
#         LCOV_INPUTS+=" --add-tracefile $PWD/$1.txt"
#     fi
# }
# By default no flags are passed to genn-buildmodel.sh or make
BUILD_FLAGS=""
MAKE_FLAGS=""
REPORT=0

# Parse command line arguments
OPTIND=1
while getopts "crd" opt; do
    case "$opt" in
    c)  BUILD_FLAGS="-c"
        MAKE_FLAGS="CPU_ONLY=1"
        ;;
    r) REPORT=1
        ;;
    d) source /opt/rh/devtoolset-6/enable
        ;;
    esac
done

# Find this script i.e. tests directory and hence GeNN itself
TESTS_DIR=$(dirname "$0")
GENN_PATH=$TESTS_DIR/../

# Clean GeNN library
pushd $GENN_PATH
make clean COVERAGE=1
popd

# Push tests directory
pushd $TESTS_DIR

# Loop through feature tests
for f in features/* ; do
    echo "Running test $f..."

    # Push feature directory
    pushd $f

    # Reset coverage  before running test
    #reset_coverage

    # Determine where the sim code is located for this test
    c=$(basename $f)"_CODE"

    # Clean test 
    # **NOTE** we do this to be sure profile data is deleted, even if building model fails
    make clean SIM_CODE=$c
    
    # Run code generator once, generating coverage
    if genn-buildmodel.sh $BUILD_FLAGS -v model.cc; then
        # Build test
        if make SIM_CODE=$c; then
            # Run tests
            ./test --gtest_output="xml:test_results$s.xml"
        fi
    fi

    # Update coverage after test
    #update_coverage coverage$s

    # Pop feature directory
    popd
done;


# # Run unit tests
# pushd unit
#
# # Reset coverage  before running test
# reset_coverage
# 
# # Clean and build
# make clean all COVERAGE=1 $MAKE_FLAGS 1>>../msg 2>>../msg 
# 
# # Run tests
# ./test --gtest_output="xml:test_results_unit.xml"
# 
# # Update coverage after test
# update_coverage coverage_unit
# 
# # Pop unit tests directory
# popd
# 
# # Run SpineML tests
# pushd spineml
# pushd simulator
# 
# # Clean and build
# make clean all $MAKE_FLAGS 1>>../../msg 2>>../../msg 
# 
# # Run SpineML simulator tests
# ./test --gtest_output="xml:test_results_spineml.xml"
# 
# 
# popd    # simulator
# popd    # spineml

if [[ "$(uname)" = "Darwin" ]]; then
    echo "Coverage not currently implemented on Mac OS X"
    
    # Loop through features and build list of raw profile output files
    for f in features/* ; do
        if [[ -f "$f/default.profraw" && -f "$f/generator_coverage" ]]; then
            LLVM_PROFRAW_FILES+="$f/default.profraw "
            
            if [ -z "$LLVM_TEST_EXECUTABLES" ]; then
                LLVM_TEST_EXECUTABLES+="$f/generator_coverage "
            else
                LLVM_TEST_EXECUTABLES+="-object $f/generator_coverage "
            fi
        fi
    done
    
    # Merge coverage
    xcrun llvm-profdata merge -sparse $LLVM_PROFRAW_FILES -o coverage.profdata
    
    # 'Show' text based coverage
    xcrun llvm-cov show $LLVM_TEST_EXECUTABLES -instr-profile=coverage.profdata > coverage_$NODE_NAME.txt
else
    # Loop through directories in which there might be coverage
    for OBJ_DIR in ${GENN_PATH}obj_coverage/*/ ; do
        # Get corresponding module name
        MODULE=$(basename $OBJ_DIR)

        # Use lcov to capture all coverage for this module
        lcov --directory $OBJ_DIR --base-directory ${GENN_PATH}src/$MODULE/ --capture -rc lcov_branch_coverage=1 --output-file ${MODULE}_coverage.txt

        # Add tracefile to list of tracefile arguments to pass to lcov
        LCOV_TRACEFILE_ARGS+=" --add-tracefile ${MODULE}_coverage.txt" 
    done

    # Combine all tracefiles together
    lcov $LCOV_TRACEFILE_ARGS --output-file coverage_$NODE_NAME.txt 

    # Strip system libraries from output
    lcov --remove coverage_$NODE_NAME.txt "/usr/*" --output-file coverage_$NODE_NAME.txt
fi

if [ $REPORT -eq 1 ]; then
    echo "Generating HTML coverage report..."

    # Generate browseable HTML
    genhtml coverage.txt --branch-coverage --output-directory ./code_coverage_report/ 1>> ../../msg 2>> msg
fi

popd    # tests
