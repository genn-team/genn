#!/bin/bash
# By default no flags are passed to genn-buildmodel.sh
BUILD_FLAGS=""
REPORT=0

# Parse command line arguments
OPTIND=1
while getopts "cr" opt; do
    case "$opt" in
    c)  BUILD_FLAGS="-c"
        ;;
    r) REPORT=1
        ;;
    esac
done

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Delete existing output
rm -f msg
rm -rf *.gcno *.gcda

# Loop through feature tests
LCOV_INPUTS="";
for f in features/*;
    do
        echo "Calculating code generation coverage of $f..."

        # Push feature directory
        pushd $f
    
        # Loop through model suffixes
        for s in "" _new;
            do
                if [ -f "model$s.cc" ]; then
                    # On OSX remove existing raw coverage files before running each test
                    # **NOTE** GCC can successfully combine gcno and gcda files itself but not LLVM
                    if [[ "$(uname)" = "Darwin" ]]; then
                        rm -f *.gcno *.gcda
                        rm -rf $GENN_PATH/lib/**/*.gcno $GENN_PATH/lib/**/*.gcda
                    fi

                    # Clean
                    make clean 1>> ../../msg 2>> ../../msg

                    # Build and generate model (measuring coverage)
                    genn-buildmodel.sh $BUILD_FLAGS -v model$s.cc  1>> ../../msg 2>>../../msg || exit $?

                    # On OSX convert the coverage of each test to LCOV format
                    if [[ "$(uname)" = "Darwin" ]]; then
                        # Capture GCOV output for this test
                        lcov --directory $GENN_PATH --capture -rc lcov_branch_coverage=1 --output-file coverage$s.txt 1>> ../../msg 2>> ../../msg

                        # Add this test's output to LCOV command line
                        LCOV_INPUTS+=" --add-tracefile $PWD/coverage$s.txt"
                    fi
                fi
            done;

        # Pop feature directory
        popd
    done;

echo "Combining coverage data..."

# On OSX combine the LCOV output gathered for each test into a single report
if [[ "$(uname)" = "Darwin" ]]; then
    lcov $LCOV_INPUTS --output-file coverage.txt 
# Otherwise generate report from raw GCOV output in child directories
else
    lcov --directory $GENN_PATH --capture -rc lcov_branch_coverage=1 --output-file coverage.txt  1>> ../../msg 2>> msg
fi

# Remove standard library stuff from coverage report
lcov --remove coverage.txt "/usr*" --output-file coverage.txt

# Remove coverage of tests themselves as this seems dumb
lcov --remove coverage.txt "tests*" --output-file coverage.txt

if [ $REPORT -eq 1 ]; then
  echo "Generating HTML coverage report..."

  # Generate browseable HTML
  genhtml coverage.txt --branch-coverage --output-directory ./code_coverage_report/ 1>> ../../msg 2>> msg
fi