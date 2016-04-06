#!/bin/bash

echo "warning: buildmodel.sh has been depreciated!"
echo "please use the new genn-buildmodel.sh script in future"

error() {
    local message="$1"
    local code="${2:-1}"
    if [[ -n "$message" ]] ; then
	echo "buildmodel.sh: Error: ${message}; exiting with status ${code}"
    else
	echo "buildmodel.sh: Error: exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ' ERR

MODELPATH=$(pwd);
MODELNAME=$1;
if [[ $MODELNAME == "" ]]; then
    echo "buildmodel.sh: Error: No arguments given"
    exit 1
fi
echo "model path:" $MODELPATH
echo "model name:" $MODELNAME

k=0
DBGMODE=0
EXTRA_DEF=
for op in $@; do
    if [[ $k > 0 ]]; then
	op=$(echo $op | tr [a-z] [A-Z])
	if [[ $op == "DEBUG=1" ]]; then
	    DBGMODE=1
	fi
	if [[ $op == "CPU_ONLY=1" ]]; then
	    EXTRA_DEF=CPU_ONLY
	fi
    fi
    k=$[$k+1];
done
if [[ $EXTRA_DEF != "" ]]; then
    EXTRA_DEF=-D$EXTRA_DEF
fi

if [[ "$GENN_PATH" == "" ]]; then
    if [[ $(uname -s) == "Linux" ]]; then
	echo "GENN_PATH is not defined. Auto-detecting..."
	export GENN_PATH=$(readlink -f $(dirname $0)/../..)
    else
	echo "buildmodel.sh: Error: GENN_PATH is not defined"
	exit 1	
    fi
fi

make clean -f "$GENN_PATH/lib/src/GNUmakefile"
if [[ "$DBGMODE" == "1" ]]; then
    echo "debugging mode ON"
    make debug -f "$GENN_PATH/lib/src/GNUmakefile" MODEL=$MODELPATH/$MODELNAME.cc EXTRA_DEF=$EXTRA_DEF;
    gdb -tui --args ./generateALL $MODELPATH;
else
    make -f "$GENN_PATH/lib/src/GNUmakefile" MODEL=$MODELPATH/$MODELNAME.cc EXTRA_DEF=$EXTRA_DEF;
    ./generateALL $MODELPATH;
fi

echo "Model build complete ..."
