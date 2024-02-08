#! /bin/bash
python mkdoc.py -o docStrings.h -std=c++17 -I ../../include/genn/third_party ../../include/genn/genn/*.h
