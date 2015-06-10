#! /bin/bash

projects="HHVclampGA_project Izh_sparse_project MBody1_project MBody_delayedSyn_project MBody_individualID_project MBody_userdef_project Model_Schmuker_2014_classifier_project OneComp_project PoissonIzh_project SynDelay_project"

for i in $projects; do
    cd $i
    make clean
    rm -rf *.dSYM/
    rm -f *.o 
    if [ -f model ]; then
	cd model
	make purge
	cd ..
    fi
    cd ..
done
