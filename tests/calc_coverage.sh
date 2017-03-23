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

# Loop through feature tests
for f in features/*;
    do
        echo "Calculating code generation coverage of $f..."

        # Push feature directory
        pushd $f
    
        # Loop through model suffixes
        for s in "" _new;
            do
                # Clean
                make clean 1>> ../../msg 2>> ../../msg

                # Build and generate model (measuring coverage)
                genn-buildmodel.sh $BUILD_FLAGS -v model$s.cc  1>> ../../msg 2>>../../msg || exit $?
            done;

        # Pop feature directory
        popd
    done;

echo "Combining coverage data..."

# Combine all GCOV ouratput in child directories
lcov --directory $GENN_PATH --capture --output-file coverage.txt -rc lcov_branch_coverage=1 1>> ../../msg 2>> msg

# Remove standard library stuff from coverage report
lcov --remove coverage.txt "/usr*" -o coverage.txt

# Remove coverage of tests themselves as this seems dumb
lcov --remove coverage.txt "tests*" -o coverage.txt

if [ $REPORT -eq 1 ]; then
  echo "Generating HTML coverage report..."

  # Generate browseable HTML
  genhtml coverage.txt --branch-coverage --output-directory ./code_coverage_report/ 1>> ../../msg 2>> msg
fi