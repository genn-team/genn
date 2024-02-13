#! /bin/bash
python mkdoc.py -o docStrings.h -std=c++17 -I ../../include/genn/third_party ../../include/genn/genn/*.h

if [[ -n "$CUDA_PATH" ]]; then
  python mkdoc.py -o cudaBackendDocStrings.h -std=c++17 -I ../../include/genn/third_party -I ../../include/genn/third_party/plog -I ../../include/genn/genn/ -I $CUDA_PATH/include ../../include/genn/backends/cuda/*.h
fi

