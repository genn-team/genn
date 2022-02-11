#! /bin/bash
export GENN_PATH=$(dirname $(realpath "$0"))
(cat doxygen/genn-doxygen.conf  ; echo "PROJECT_NUMBER=`cat version.txt`") | doxygen -
