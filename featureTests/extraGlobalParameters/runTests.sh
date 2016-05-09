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

echo \# building EGSPInSimCodeEvnt_sparseInv
buildmodel.sh EGSPInSimCodeEvnt_sparseInv &>msg
make -f MakefileEGSPInSimCodeEvnt_sparseInv clean &>msg
make -f MakefileEGSPInSimCodeEvnt_sparseInv &>msg
echo \#-----------------------------------------------------------
echo \# running testEGSPInSimCodeEvnt_sparseInv on GPU ...
./testEGSPInSimCodeEvnt_sparseInv 1 test1 0 
echo \# running testEGSPInSimCodeEvnt_sparseInv on CPU ...
./testEGSPInSimCodeEvnt_sparseInv 0 test1CPU 0 
