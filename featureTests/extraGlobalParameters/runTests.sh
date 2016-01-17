#! /bin/bash

echo \# building EGPInSimCode
buildmodel.sh EGPInSimCode &>msg
make -f MakefileEGPInSimCode clean &>msg
make -f MakefileEGPInSimCode &>msg
echo \#-----------------------------------------------------------
echo \# running testEGPInSimCode on GPU ...
./testEGPInSimCode 1 test1 0 
echo \# running testEGPInSimCode on CPU ...
./testEGPInSimCode 0 test1CPU 0 
