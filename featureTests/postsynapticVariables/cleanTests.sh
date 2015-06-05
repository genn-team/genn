#! /bin/bash

make -f MakefilePostVarsInSimCode clean
make -f MakefilePostVarsInSynapseDynamics clean
make -f MakefilePostVarsInPostLearn clean
make -f MakefilePostVarsInSimCode_sparse clean
make -f MakefilePostVarsInSynapseDynamics_sparse clean
make -f MakefilePostVarsInPostLearn_sparse clean
rm -rf *_CODE
rm msg
rm sm_version.mk
rm *.dat
