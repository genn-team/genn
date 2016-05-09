#! /bin/bash

make -f MakefileEGPInSimCode clean
make -f MakefileEGSPInSimCodeEvnt_sparseInv clean
rm -rf *_CODE
rm -f msg
rm -f sm_version.mk
rm -f *.dat
rm -f generateALL
rm -f runner.cubin
