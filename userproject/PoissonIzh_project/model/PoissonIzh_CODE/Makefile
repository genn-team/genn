
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-c -std=c++11 -x cu -arch sm_30  -O3 -use_fast_math -Xcompiler "-ffast-math"

INCLUDEFLAGS   =-I"$(GENN_PATH)/lib/include"

all: runner.o

runner.o: runner.cc
	$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc

clean:
	rm -f runner.o
