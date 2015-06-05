#! /bin/bash

echo \# building postVarsInSimCode
buildmodel.sh postVarsInSimCode &>msg
make -f MakefilePostVarsInSimCode clean &>msg
make -f MakefilePostVarsInSimCode &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSimCode on GPU ...
./testPostVarsInSimCode 1 test1 0 
echo \# running testPostVarsInSimCode on CPU ...
./testPostVarsInSimCode 0 test1CPU 0 

echo \# building postVarsInSynapseDynamics
buildmodel.sh postVarsInSynapseDynamics &>msg
make -f MakefilePostVarsInSynapseDynamics clean &>msg
make -f MakefilePostVarsInSynapseDynamics &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSynapseDynamics on GPU ...
./testPostVarsInSynapseDynamics 1 test2 0 
echo \# running testPostVarsInSynapseDynamics on CPU ...
./testPostVarsInSynapseDynamics 0 test2CPU 0 

echo \# building postVarsInPostLearn
buildmodel.sh postVarsInPostLearn &>msg
make -f MakefilePostVarsInPostLearn clean &>msg
make -f MakefilePostVarsInPostLearn &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInPostLearn on GPU ...
./testPostVarsInPostLearn 1 test3 0 
echo \# running testPostVarsInPostLearn on CPU ...
./testPostVarsInPostLearn 0 test3CPU 0 

echo \# building postVarsInSimCode_sparse
buildmodel.sh postVarsInSimCode_sparse &>msg
make -f MakefilePostVarsInSimCode_sparse clean &>msg
make -f MakefilePostVarsInSimCode_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSimCode_sparse on GPU ...
./testPostVarsInSimCode_sparse 1 test4 0 
echo \# running testPostVarsInSimCode_sparse on CPU ...
./testPostVarsInSimCode_sparse 0 test4CPU 0 

echo \# building postVarsInSynapseDynamics_sparse
buildmodel.sh postVarsInSynapseDynamics_sparse &>msg
make -f MakefilePostVarsInSynapseDynamics_sparse clean &>msg
make -f MakefilePostVarsInSynapseDynamics_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInSynapseDynamics_sparse on GPU ...
./testPostVarsInSynapseDynamics_sparse 1 test5 0 
echo \# running testPostVarsInSynapseDynamics_sparse on CPU ...
./testPostVarsInSynapseDynamics_sparse 0 test5CPU 0 

echo \# building postVarsInPostLearn_sparse
buildmodel.sh postVarsInPostLearn_sparse &>msg
make -f MakefilePostVarsInPostLearn_sparse clean &>msg
make -f MakefilePostVarsInPostLearn_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPostVarsInPostLearn_sparse on GPU ...
./testPostVarsInPostLearn_sparse 1 test5 0 
echo \# running testPostVarsInPostLearn_sparse on CPU ...
./testPostVarsInPostLearn_sparse 0 test5CPU 0 

