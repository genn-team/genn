#!/bin/bash


reset_coverage () {
    # On OSX remove existing raw coverage files before running each test
    # **NOTE** GCC can successfully combine gcno and gcda files itself but not LLVM
    # **NOTE** only remove libgenn coverage COUNTS as libgenn itself and hence its .gcdo files don't get rebuilt
    if [[ "$(uname)" = "Darwin" ]]; then
        rm -f *.gcno *.gcda
        rm -rf $GENN_PATH/lib/**/*.gcda
    fi
}

update_coverage () {
    # On OSX convert the coverage of each test to LCOV format
    if [[ "$(uname)" = "Darwin" ]]; then
        # Capture GCOV output for this test
        lcov --directory $GENN_PATH --capture -rc lcov_branch_coverage=1 --output-file $1.txt 1>> ../../msg 2>> ../../msg

        # Add this test's output to LCOV command line
        LCOV_INPUTS+=" --add-tracefile $PWD/$1.txt"
    fi
}
# By default no flags are passed to genn-buildmodel.sh or make
BUILD_FLAGS=""
MAKE_FLAGS=""
REPORT=0

# Parse command line arguments
OPTIND=1
while getopts "cr" opt; do
    case "$opt" in
    c)  BUILD_FLAGS="-c"
        MAKE_FLAGS="CPU_ONLY=1"
        ;;
    r) REPORT=1
        ;;
    esac
done

# Clean GeNN library
pushd $GENN_PATH/lib
make clean
popd

# Delete existing output and libgenn coverage data
rm -f msg
rm -rf $GENN_PATH/lib/**/*.gcno $GENN_PATH/lib/**/*.gcda

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
                    # Reset coverage  before running test
                    reset_coverage
                    
                    # Determine where the sim code is located for this test
                    c=$(basename $f)$s"_CODE"

                    # Clean
                    make $MAKE_FLAGS SIM_CODE=$c clean 1>> ../../msg 2>> ../../msg

                    # Build and generate model (generating coverage)
                    if genn-buildmodel.sh $BUILD_FLAGS -v model$s.cc 1>>../../msg 2>> ../../msg ; then
                        # Make
                        if make $MAKE_FLAGS SIM_CODE=$c 1>>../../msg 2>>../../msg ; then
                            # Run tests
                            ./test --gtest_output="xml:test_results$s.xml"
                        fi
                    fi
                    
                    # Update coverage after test
                    update_coverage coverage$s
                fi
            done;

        # Pop feature directory
        popd
    done;

# Run unit tests
pushd unit

# Reset coverage  before running test
reset_coverage

# Clean and build
make clean all COVERAGE=1 $MAKE_FLAGS 1>>../msg 2>>../msg 

# Run tests
./test --gtest_output="xml:test_results_unit.xml"

# Update coverage after test
update_coverage coverage_unit

# Pop unit tests directory
popd

# Run SpineML tests
pushd spineml
pushd simulator

# Clean and build
make clean all $MAKE_FLAGS 1>>../../msg 2>>../../msg 

# Run SpineML simulator tests
./test --gtest_output="xml:test_results_spineml.xml"


popd    # simulator
popd    # spineml

echo "Combining coverage data..."

# On OSX combine the LCOV output gathered for each test into a single report
if [[ "$(uname)" = "Darwin" ]]; then
    lcov $LCOV_INPUTS --output-file coverage.txt 
# Otherwise generate report from raw GCOV output in child directories
else
    lcov --directory $GENN_PATH --capture -rc lcov_branch_coverage=1 --output-file coverage.txt  1>> ../../msg 2>> msg
fi

if [ $REPORT -eq 1 ]; then
    echo "Generating HTML coverage report..."

    # Do preprocessing normally performed by Jenkins
    lcov --remove coverage.txt "/usr/*" --output-file coverage.txt
    lcov --remove coverage.txt "*/test/*" --output-file coverage.txt
    
    # Generate browseable HTML
    genhtml coverage.txt --branch-coverage --output-directory ./code_coverage_report/ 1>> ../../msg 2>> msg
fi