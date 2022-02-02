#! /bin/bash
export GENN_PATH=/its/home/tn41/localdisk_projects/develop/genn
(cat doxygen/genn-doxygen.conf  ; echo "PROJECT_NUMBER=`cat version.txt`") | doxygen -
