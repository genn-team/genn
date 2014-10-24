#!/bin/bash

# BASH SCRIPT TO LINK SPINEML TO GENN
# ALEX COPE - 2013
# UNIVERSITY OF SHEFFIELD

#exit on first error
set -e

# What OS are we?
if [ $(uname) = 'Linux' ]; then
    OS='Linux'
elif [ $(uname) = 'Windows_NT' ] || [ $(uname) = 'MINGW32_NT-6.1' ]; then
    OS='Windows'
else
    OS='OSX'
fi

echo ""
echo "Converting SpineML to GeNN"
echo "Alex Cope             2013"
echo "##########################"
echo ""
echo "Creating extra_neurons.h file with new neuron_body components..."
xsltproc -o extra_neurons.h SpineML_2_GeNN_neurons.xsl model/experiment.xml
echo "Done"
echo "Creating extra_postsynapses.h file with new postsynapse components..."
xsltproc -o extra_postsynapses.h SpineML_2_GeNN_postsynapses.xsl model/experiment.xml
echo "Done"
echo "Creating extra_weightupdates.h file with new weightupdate components..."
xsltproc -o extra_weightupdates.h SpineML_2_GeNN_weightupdates.xsl model/experiment.xml
echo "Done"
echo "Creating model.cc file..."
xsltproc -o model.cc SpineML_2_GeNN_model.xsl model/experiment.xml
echo "Done"
echo "Creating sim.cu file..."
xsltproc -o sim.cu SpineML_2_GeNN_sim.xsl model/experiment.xml
echo "Done"
echo "Running GeNN code generation..."
if [[ -z ${GENN_PATH+x} ]]; then
echo "Sourcing .bashrc as environment does not seem to be correct"
source ~/.bashrc
fi
if [[ -z ${GENN_PATH+x} ]]; then
error_exit "The system environment is not correctly configured"
fi
cp extra_neurons.h $GENN_PATH/lib/include/
cp extra_postsynapses.h $GENN_PATH/lib/include/
cp extra_weightupdates.h $GeNNPATH/lib/include/
cp sim.cu $GENN_PATH/userproject/model_project/sim.cu
if cp model/*.bin $GENN_PATH/userproject/model_project/; then
	echo "Copying binary data..."	
fi
cd $GENN_PATH/userproject/model_project
../../lib/bin/buildmodel model $DBGMODE
make clean
make

if [ $OS = 'Linux' ]; then
if mv *.bin bin/linux/release/; then
	echo "Moving binary data..."	
fi
cd bin/linux/release
fi
if [ $OS = 'OSX' ]; then
if mv *.bin bin/darwin/release/; then
	echo "Moving binary data..."	
fi
cd bin/darwin/release
fi
./sim
#rm *.bin
echo "Done"
echo ""
echo "Finished"
