#! /bin/bash

make -f MakefilePreVarsInSimCode clean
make -f MakefilePreVarsInSimCodeEvnt clean
make -f MakefilePreVarsInSynapseDynamics clean
make -f MakefilePreVarsInPostLearn clean
make -f MakefilePreVarsInSimCode_sparse clean
make -f MakefilePreVarsInSimCodeEvnt_sparse clean
make -f MakefilePreVarsInSimCodeEvnt_sparseInv clean
make -f MakefilePreVarsInSynapseDynamics_sparse clean
make -f MakefilePreVarsInPostLearn_sparse clean
rm -rf *_CODE
rm -f msg
rm -f sm_version.mk
rm -f *.dat
rm -f generateALL
rm -f runner.cubin
