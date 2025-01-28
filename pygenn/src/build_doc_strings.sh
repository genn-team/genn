#! /bin/bash
python mkdoc.py -o docStrings.h -std=c++17 -I ../../include/genn/third_party -I ../../include/genn/genn/ ../../include/genn/genn/*.h ../../include/genn/genn/code_generator/backendBase.h ../../include/genn/genn/code_generator/backendCUDAHIP.h

if [[ -n "$CUDA_PATH" ]]; then
  python mkdoc.py -o cudaBackendDocStrings.h -std=c++17 -I ../../include/genn/third_party -I ../../include/genn/third_party/plog -I ../../include/genn/genn/ -I $CUDA_PATH/include ../../include/genn/backends/cuda/*.h
fi

if [[ -n "$HIP_PATH" ]]; then
  if [ "$HIP_PLATFORM" == "nvidia" ]; then
    python mkdoc.py -o hipBackendDocStrings.h -std=c++17 -D__HIP_PLATFORM_NVIDIA__ -I ../../include/genn/third_party -I ../../include/genn/third_party/plog -I ../../include/genn/genn/ -I $HIP_PATH/include -I $CUDA_PATH/include ../../include/genn/backends/hip/*.h
  else
    python mkdoc.py -o hipBackendDocStrings.h -std=c++17 -D__HIP_PLATFORM_AMD__ -I ../../include/genn/third_party -I ../../include/genn/third_party/plog -I ../../include/genn/genn/ -I $HIP_PATH/include ../../include/genn/backends/hip/*.h
  fi
fi
