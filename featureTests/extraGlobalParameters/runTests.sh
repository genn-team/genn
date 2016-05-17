#! /bin/bash

echo \# building EGPInSimCode
genn-buildmodel.sh EGPInSimCode.cc &>msg
make -f MakefileEGPInSimCode SIM_CODE=EGPInSimCode_CODE clean &>msg
make -f MakefileEGPInSimCode SIM_CODE=EGPInSimCode_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testEGPInSimCode on GPU ...
./testEGPInSimCode 1 test1 0 
echo \# running testEGPInSimCode on CPU ...
./testEGPInSimCode 0 test1CPU 0 

echo \# building EGSPInSimCodeEvnt_sparseInv
genn-buildmodel.sh EGSPInSimCodeEvnt_sparseInv.cc &>msg
make -f MakefileEGSPInSimCodeEvnt_sparseInv SIM_CODE=EGSPInSimCodeEvnt_sparseInv_CODE clean &>msg
make -f MakefileEGSPInSimCodeEvnt_sparseInv SIM_CODE=EGSPInSimCodeEvnt_sparseInv_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testEGSPInSimCodeEvnt_sparseInv on GPU ...
./testEGSPInSimCodeEvnt_sparseInv 1 test1 0 
echo \# running testEGSPInSimCodeEvnt_sparseInv on CPU ...
./testEGSPInSimCodeEvnt_sparseInv 0 test1CPU 0 
