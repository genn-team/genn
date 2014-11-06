#!/bin/bash

# BASH SCRIPT TO LINK SPINEML TO GENN
# ALEX COPE - 2013
# UNIVERSITY OF SHEFFIELD

#exit on first error
set -e

# get the command line options...
while getopts w:srm:o:n:a:vV\? opt
do
case "$opt" in
w)  GENN_2_BRAHMS_DIR="$OPTARG"
;;
s)  REBUILD_SYSTEMML="false"
;;
r)  REBUILD_COMPONENTS="true"
;;
m)  MODEL_DIR="$OPTARG"
;;
o)  OUTPUT_DIR="$OPTARG"
;;
n)  NODES="$OPTARG"
;;
a)  NODEARCH="$OPTARG"
;;
v)  VERBOSE_BRAHMS="--d"
;;
V)  VERBOSE_BRAHMS="--dd"
;;
\?) usage
;;
esac
done
shift `expr $OPTIND - 1`

# What OS are we?
if [ $(uname) = 'Linux' ]; then
if [ $(uname -i) = 'i686' ]; then
OS='Linux'
else
OS='Linux'
fi
elif [ $(uname) = 'Windows_NT' ] || [ $(uname) = 'MINGW32_NT-6.1' ]; then
OS='Windows'
else
OS='OSX'
fi

echo ""
echo "Converting SpineML to GeNN"
echo "Alex Cope             2014"
echo "##########################"
echo ""
echo "Creating extra_neurons.h file with new neuron_body components..."
xsltproc -o extra_neurons.h SpineML_2_GeNN_neurons.xsl model/experiment.xml
echo "Done"
echo "Creating extra_postsynapses.h file with new postsynapse components..."
xsltproc -o extra_postsynapses.h SpineML_2_GeNN_postsynapses.xsl model/experiment.xml
echo "Done"
echo "Creating model.cc file..."
xsltproc -o model.cc SpineML_2_GeNN_model.xsl model/experiment.xml
echo "Done"
echo "Creating sim.cu file..."
xsltproc -o sim.cu SpineML_2_GeNN_sim.xsl model/experiment.xml
echo "Done"
echo "Running GeNN code generation..."
if [[ -z ${GeNNPATH+x} ]]; then
echo "Sourcing .bashrc as environment does not seem to be correct"
source ~/.bashrc
fi
if [[ -z ${GeNNPATH+x} ]]; then
error_exit "The system environment is not correctly configured"
fi
# update the tools
cd $GeNNPATH/tools
make
cd -
#check the directory is there
mkdir -p $GeNNPATH/userproject/model_project
cp extra_neurons.h $GeNNPATH/lib/include/
cp extra_postsynapses.h $GeNNPATH/lib/include/
cp rng.h $GeNNPATH/userproject/model_project/
cp Makefile $GeNNPATH/userproject/model_project/
cp model.cc $GeNNPATH/userproject/model_project/model.cc
cp sim.cu $GeNNPATH/userproject/model_project/sim.cu
if cp model/*.bin $GeNNPATH/userproject/model_project/; then
	echo "Copying binary data..."	
fi
cd $GeNNPATH/userproject/model_project
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
