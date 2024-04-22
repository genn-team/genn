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

# Clean GeNN library and build a version of the single-threaded CPU backend with coverage calculation built
cd $GENN_PATH

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
    # Loop through all object directories with coverage data
    for OBJ_DIR in obj_coverage*/ ; do
        # Use lcov to capture libgenn coverage
        OBJ_NAME=$(basename $OBJ_DIR)
        lcov --directory ${OBJ_DIR}/genn/genn --base-directory src/genn/genn --capture -rc lcov_branch_coverage=1 --output-file genn_${OBJ_NAME}.txt

        # Add tracefile to list of tracefile arguments to pass to lcov
        LCOV_TRACEFILE_ARGS+=" --add-tracefile genn_${OBJ_NAME}.txt" 

        # Loop through directories in which there might be coverage for backends
        for BACKEND_OBJ_DIR in ${OBJ_DIR}/genn/backends/*/ ; do
            # Get corresponding module name
            MODULE=$(basename $BACKEND_OBJ_DIR)

            # Use lcov to capture all coverage for this module
            lcov --directory $BACKEND_OBJ_DIR --base-directory src/genn/backends/$MODULE/ --capture -rc lcov_branch_coverage=1 --output-file ${MODULE}_${OBJ_NAME}.txt

            # Add tracefile to list of tracefile arguments to pass to lcov
            LCOV_TRACEFILE_ARGS+=" --add-tracefile ${MODULE}_${OBJ_NAME}.txt" 
        done
    done

    # Combine all tracefiles together
    lcov $LCOV_TRACEFILE_ARGS --output-file coverage_$NODE_NAME.txt 

    # Strip system libraries from output
    lcov --remove coverage_$NODE_NAME.txt "/usr/*" --output-file coverage_$NODE_NAME.txt
fi

if [ $REPORT -eq 1 ]; then
    echo "Generating HTML coverage report..."

    # Generate browseable HTML
    genhtml coverage_$NODE_NAME.txt --branch-coverage --output-directory ./code_coverage_report/
fi
