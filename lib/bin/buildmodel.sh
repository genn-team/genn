#!/bin/bash

FLAGS="$PWD/$1.cc"
k=0
for op in $@; do
    if [[ $k > 0 ]]; then
	op=$(echo $op | tr [a-z] [A-Z])
	if [[ $op == "DEBUG=1" ]]; then
	    FLAGS+=" -d"
	fi
	if [[ $op == "CPU_ONLY=1" ]]; then
	    FLAGS+=" -c"
	fi
    fi
    k=$((k+1));
done

echo "warning: buildmodel.sh has been depreciated!"
echo "please use the new genn-buildmodel.sh script in future"
echo "the equivalent genn-buildmodel.sh command is:"
echo "genn-buildmodel.sh $FLAGS"

genn-buildmodel.sh $FLAGS
