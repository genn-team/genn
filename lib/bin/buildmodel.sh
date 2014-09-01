#!/bin/bash

MODELPATH=$(pwd);
echo "model path:" $MODELPATH
MODELNAME=$1;
echo "model name:" $MODELNAME
DBGMODE=$2; # 1 if debugging, 0 if release

cp $MODELPATH/$MODELNAME.cc $GENNPATH/lib/src/currentModel.cc;
cd $GENNPATH/lib;

make clean
if [ "$DBGMODE" = "1" ]
then
	echo "debugging mode ON"
	make debug;
	gdb -tui --directory=$GENNPATH/lib/bin --args generateALL $MODELPATH;
else
	make;
	bin/generateALL $MODELPATH;
fi
cd $MODELPATH;

echo "Model build complete ..."
