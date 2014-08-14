#! /bin/bash
export GeNNMODELNAME=$1;
if [[ -n "$COMSPEC" ]]; then #This is to check if we are on a Windows environment
	export WD=$(cygpath -w $(pwd))
else
	export WD=$(pwd)
fi
export DBGMODE=$2; # 1 if debugging, 0 if release
echo "genn model name:" $1
echo "wd: " $WD
export GeNNMODELINCLUDE=$WD/$GeNNMODELNAME.cc\ 
cp $GeNNMODELINCLUDE $GeNNPATH/lib/src/currentModel.cc
echo " GeNNMODELINCLUDE" = $GeNNMODELINCLUDE
cd $GeNNPATH/lib;

if [ "$DBGMODE" = "1" ]
then
	#debugging: type "break main" or other breakpoint and then "run" once you are in cuda-gdb
	echo "debugging mode ON"
	make clean && make debug;
	gdb -tui --directory=$GeNNPATH/lib --args bin/generateALL $WD;
else
	make clean && make;
	bin/generateALL $WD;
fi
echo " Model build complete ..." 
cd $WD;
