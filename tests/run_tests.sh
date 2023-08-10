#!/bin/bash
# By default no flags are passed to genn-buildmodel.sh
REPORT=0

# Parse command line arguments
OPTIND=1
while getopts "r" opt; do
    case "$opt" in
    r) REPORT=1
        ;;
    esac
done

# Find this script i.e. tests directory and hence GeNN itself
TESTS_DIR=$(dirname "$0")
GENN_PATH=$TESTS_DIR/../

# Count cores using approach lifted from https://stackoverflow.com/questions/6481005/how-to-obtain-the-number-of-cpus-cores-in-linux-from-the-command-line
if [[ $(uname) = "Darwin" ]]; then
    CORE_COUNT=$(sysctl -n hw.physicalcpu_max)
else
    CORE_COUNT=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

# Clean GeNN library
pushd $GENN_PATH
make clean COVERAGE=1
popd

# Make sure we have a version of the single-threaded CPU backend with coverage calculation built
pushd $GENN_PATH
make single_threaded_cpu -j $CORE_COUNT COVERAGE=1
popd

# Run unit tests
pushd unit

# Clean and build
make clean all -j $CORE_COUNT COVERAGE=1

# Run tests
./test_coverage --gtest_output="xml:test_results_unit.xml"

popd    # unit

if [[ "$(uname)" = "Darwin" ]]; then
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
    
    # Add unit tests profiling data to lists
    if [[ -f "unit/default.profraw" && -f "unit/test_coverage" ]]; then
            LLVM_PROFRAW_FILES+="unit/default.profraw "
            
            if [ -z "$LLVM_TEST_EXECUTABLES" ]; then
                LLVM_TEST_EXECUTABLES+="unit/test_coverage "
            else
                LLVM_TEST_EXECUTABLES+="-object unit/test_coverage "
            fi
        fi
        
    # Merge coverage
    xcrun llvm-profdata merge -sparse $LLVM_PROFRAW_FILES -o coverage.profdata
    
    # 'Show' text based coverage
    xcrun llvm-cov show $LLVM_TEST_EXECUTABLES -instr-profile=coverage.profdata > coverage_$NODE_NAME.txt
else
    # Use lcov to capture libgenn coverage
    lcov --directory ${GENN_PATH}obj_coverage/genn/genn --base-directory ${GENN_PATH}src/genn/genn --capture -rc lcov_branch_coverage=1 --output-file genn_coverage.txt

    # Add tracefile to list of tracefile arguments to pass to lcov
    LCOV_TRACEFILE_ARGS+=" --add-tracefile genn_coverage.txt" 
    
    # Loop through directories in which there might be coverage for backends
    for BACKEND_OBJ_DIR in ${GENN_PATH}obj_coverage/genn/backends/*/ ; do
        # Get corresponding module name
        MODULE=$(basename $BACKEND_OBJ_DIR)

        # Use lcov to capture all coverage for this module
        lcov --directory $BACKEND_OBJ_DIR --base-directory ${GENN_PATH}src/genn/backends/$MODULE/ --capture -rc lcov_branch_coverage=1 --output-file ${MODULE}_coverage.txt

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
