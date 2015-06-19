#!/bin/bash

error() {
  local message="$1"
  local code="${2:-1}"
  if [[ -n "$message" ]] ; then
    echo "buildmodel Error: ${message}; exiting with status ${code}"
  else
    echo "buildmodel Error: exiting with status ${code}"
  fi
  exit "${code}"
}
trap 'error ' ERR

MODELPATH=$(pwd);
echo "model path:" $MODELPATH
MODELNAME=$1;
echo "model name:" $MODELNAME
DBGMODE=$2; # 1 if debugging, 0 if release

if [ "$GENN_PATH" = "" ]; then
    if [ "$GeNNPATH" = "" ]; then
	echo "buildmodel Error: Environment variable 'GENN_PATH' has not been defined. Quitting..."
	exit 1
    fi
    echo "Environment variable 'GeNNPATH' will be replaced by 'GENN_PATH' in future GeNN releases."
    export GENN_PATH=$GeNNPATH
fi

cd $GENN_PATH/lib;
make clean
if [ "$DBGMODE" = "1" ]; then
    echo "debugging mode ON"
    make debug MODEL=$MODELPATH/$MODELNAME.cc;
    gdb -tui --directory=bin --args generateALL $MODELPATH;
else
    make MODEL=$MODELPATH/$MODELNAME.cc;
    bin/generateALL $MODELPATH;
fi
cd $MODELPATH;

echo "Model build complete ..."
