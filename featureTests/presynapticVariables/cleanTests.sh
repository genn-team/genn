#! /bin/bash

make -f MakefilePreVarsInSimCode clean
make -f MakefilePreVarsInSynapseDynamics clean
make -f MakefilePreVarsInPostLearn clean
make -f MakefilePreVarsInSimCode_sparse clean
make -f MakefilePreVarsInSynapseDynamics_sparse clean
make -f MakefilePreVarsInPostLearn_sparse clean
rm -rf *_CODE
rm msg
rm sm_version.mk
rm *.dat
