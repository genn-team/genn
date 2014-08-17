#!/bin/bash

MODELPATH=$(pwd);
echo "model path:" $MODELPATH
MODELNAME=$1;
echo "model name:" $MODELNAME
DBGMODE=$2; # 1 if debugging, 0 if release

cp $MODELPATH/$MODELNAME.cc $GeNNPATH/lib/src/currentModel.cc;
cd $GeNNPATH/lib;
if [ "$DBGMODE" = "1" ]
then
	# debugging: type "break main" or other breakpoint and then "run" once you are in cuda-gdb
	echo "debugging mode ON"
	make clean && make debug;
	gdb -tui --directory=$GeNNPATH/lib/bin --args generateALL $MODELPATH;
else
	make clean && make;
	bin/generateALL $MODELPATH;
fi
cd $MODELPATH;

echo "Model build complete ..."
