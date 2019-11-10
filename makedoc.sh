#! /bin/bash
export GENN_PATH=$(dirname $(realpath "$0"))
(cat doxygen/genn-doxygen.conf  ; echo "PROJECT_NUMBER=`cat version.txt`") | doxygen -
doxygen/fixLatexPageref.pl documentation/latex/files.tex
doxygen/fixLatexPageref.pl documentation/latex/hierarchy.tex
doxygen/fixLatexPageref.pl documentation/latex/annotated.tex
doxygen/fixLatexLabels.pl documentation/latex/refman.tex
doxygen/replace_in_files.pl "\\\\backmatter" "" documentation/latex/refman.tex
