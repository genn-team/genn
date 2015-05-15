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
./testPreVarsInSynapseDynamics 1 test1 0 
echo \# running testPreVarsInSynapseDynamics on CPU ...
./testPreVarsInSynapseDynamics 0 test1CPU 0 
