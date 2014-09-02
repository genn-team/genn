#!/bin/bash

MODELPATH=$(pwd);
echo "model path:" $MODELPATH
MODELNAME=$1;
echo "model name:" $MODELNAME
DBGMODE=$2; # 1 if debugging, 0 if release

if [ "$GENN_PATH" = "" ]; then
    if [ "GeNNPATH" = "" ]; then
	echo "ERROR: Environment variable 'GENN_PATH' has not been defined. Quitting..."
	exit
    fi
    echo "Environment variable 'GeNNPATH' will be replaced by 'GENN_PATH' in future GeNN releases."
    export GENN_PATH=$GeNNPATH
fi

cp $MODELPATH/$MODELNAME.cc $GENN_PATH/lib/src/currentModel.cc;
cd $GENN_PATH/lib;

make clean
if [ "$DBGMODE" = "1" ]; then
    echo "debugging mode ON"
    make debug;
    gdb -tui --directory=bin --args generateALL $MODELPATH;
else
    make;
    bin/generateALL $MODELPATH;
fi

cd $MODELPATH;
echo "Model build complete ..."
