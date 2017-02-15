#! /bin/bash

echo \# building postVarsInSimCode
genn-buildmodel.sh postVarsInSimCode.cc &>msg
make -f MakefilePostVarsInSimCode SIM_CODE=postVarsInSimCode_CODE clean &>msg
make -f MakefilePostVarsInSimCode SIM_CODE=postVarsInSimCode_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSimCode on GPU ...
./testPostVarsInSimCode 1 test1 0 
echo \# running testPostVarsInSimCode on CPU ...
./testPostVarsInSimCode 0 test1CPU 0 

echo \# building postVarsInSynapseDynamics
genn-buildmodel.sh postVarsInSynapseDynamics.cc &>msg
make -f MakefilePostVarsInSynapseDynamics SIM_CODE=postVarsInSynapseDynamics_CODE clean &>msg
make -f MakefilePostVarsInSynapseDynamics SIM_CODE=postVarsInSynapseDynamics_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSynapseDynamics on GPU ...
./testPostVarsInSynapseDynamics 1 test2 0 
echo \# running testPostVarsInSynapseDynamics on CPU ...
./testPostVarsInSynapseDynamics 0 test2CPU 0 

echo \# building postVarsInPostLearn
genn-buildmodel.sh postVarsInPostLearn.cc &>msg
make -f MakefilePostVarsInPostLearn SIM_CODE=postVarsInPostLearn_CODE clean &>msg
make -f MakefilePostVarsInPostLearn SIM_CODE=postVarsInPostLearn_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInPostLearn on GPU ...
./testPostVarsInPostLearn 1 test3 0 
echo \# running testPostVarsInPostLearn on CPU ...
./testPostVarsInPostLearn 0 test3CPU 0 

echo \# building postVarsInSimCode_sparse
genn-buildmodel.sh postVarsInSimCode_sparse.cc &>msg
make -f MakefilePostVarsInSimCode_sparse SIM_CODE=postVarsInSimCode_sparse_CODE clean &>msg
make -f MakefilePostVarsInSimCode_sparse SIM_CODE=postVarsInSimCode_sparse_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSimCode_sparse on GPU ...
./testPostVarsInSimCode_sparse 1 test4 0 
echo \# running testPostVarsInSimCode_sparse on CPU ...
./testPostVarsInSimCode_sparse 0 test4CPU 0 

echo \# building postVarsInSynapseDynamics_sparse
genn-buildmodel.sh postVarsInSynapseDynamics_sparse.cc &>msg
make -f MakefilePostVarsInSynapseDynamics_sparse SIM_CODE=postVarsInSynapseDynamics_sparse_CODE clean &>msg
make -f MakefilePostVarsInSynapseDynamics_sparse SIM_CODE=postVarsInSynapseDynamics_sparse_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSynapseDynamics_sparse on GPU ...
./testPostVarsInSynapseDynamics_sparse 1 test5 0 
echo \# running testPostVarsInSynapseDynamics_sparse on CPU ...
./testPostVarsInSynapseDynamics_sparse 0 test5CPU 0 

echo \# building postVarsInPostLearn_sparse
genn-buildmodel.sh postVarsInPostLearn_sparse.cc &>msg
make -f MakefilePostVarsInPostLearn_sparse SIM_CODE=postVarsInPostLearn_sparse_CODE clean &>msg
make -f MakefilePostVarsInPostLearn_sparse SIM_CODE=postVarsInPostLearn_sparse_CODE &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInPostLearn_sparse on GPU ...
./testPostVarsInPostLearn_sparse 1 test5 0 
echo \# running testPostVarsInPostLearn_sparse on CPU ...
./testPostVarsInPostLearn_sparse 0 test5CPU 0 

