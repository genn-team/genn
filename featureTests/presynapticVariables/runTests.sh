#! /bin/bash

echo \# building preVarsInSimCode
buildmodel.sh preVarsInSimCode &>msg
make -f MakefilePreVarsInSimCode clean &>msg
make -f MakefilePreVarsInSimCode &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInSimCode on GPU ...
./testPreVarsInSimCode 1 test1 0 
echo \# running testPreVarsInSimCode on CPU ...
./testPreVarsInSimCode 0 test1CPU 0 

echo \# building preVarsInSynapseDynamics
buildmodel.sh preVarsInSynapseDynamics &>msg
make -f MakefilePreVarsInSynapseDynamics clean &>msg
make -f MakefilePreVarsInSynapseDynamics &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInSynapseDynamics on GPU ...
./testPreVarsInSynapseDynamics 1 test2 0 
echo \# running testPreVarsInSynapseDynamics on CPU ...
./testPreVarsInSynapseDynamics 0 test2CPU 0 

echo \# building preVarsInPostLearn
buildmodel.sh preVarsInPostLearn &>msg
make -f MakefilePreVarsInPostLearn clean &>msg
make -f MakefilePreVarsInPostLearn &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInPostLearn on GPU ...
./testPreVarsInPostLearn 1 test3 0 
echo \# running testPreVarsInPostLearn on CPU ...
./testPreVarsInPostLearn 0 test3CPU 0 

echo \# building preVarsInSimCode_sparse
buildmodel.sh preVarsInSimCode_sparse &>msg
make -f MakefilePreVarsInSimCode_sparse clean &>msg
make -f MakefilePreVarsInSimCode_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInSimCode_sparse on GPU ...
./testPreVarsInSimCode_sparse 1 test4 0 
echo \# running testPreVarsInSimCode_sparse on CPU ...
./testPreVarsInSimCode_sparse 0 test4CPU 0 

echo \# building preVarsInSynapseDynamics_sparse
buildmodel.sh preVarsInSynapseDynamics_sparse &>msg
make -f MakefilePreVarsInSynapseDynamics_sparse clean &>msg
make -f MakefilePreVarsInSynapseDynamics_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInSynapseDynamics_sparse on GPU ...
./testPreVarsInSynapseDynamics_sparse 1 test5 0 
echo \# running testPreVarsInSynapseDynamics_sparse on CPU ...
./testPreVarsInSynapseDynamics_sparse 0 test5CPU 0 

echo \# building preVarsInPostLearn_sparse
buildmodel.sh preVarsInPostLearn_sparse &>msg
make -f MakefilePreVarsInPostLearn_sparse clean &>msg
make -f MakefilePreVarsInPostLearn_sparse &>msg
echo \#-----------------------------------------------------------
echo \# running testPreVarsInPostLearn_sparse on GPU ...
./testPreVarsInPostLearn_sparse 1 test5 0 
echo \# running testPreVarsInPostLearn_sparse on CPU ...
./testPreVarsInPostLearn_sparse 0 test5CPU 0 

