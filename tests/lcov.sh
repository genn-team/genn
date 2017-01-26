#!/bin/bash

# Combine all GCOV output in child directories
lcov --directory . --capture --output-file coverage.txt -rc lcov_branch_coverage=1

# Generate browseable HTML
genhtml coverage.txt --branch-coverage --output-directory ./code_coverage_report/
