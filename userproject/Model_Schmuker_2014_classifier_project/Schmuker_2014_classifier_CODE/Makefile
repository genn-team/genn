
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-c -x cu -arch sm_30 

INCLUDEFLAGS   =-I"$(GENN_PATH)/lib/include"

all: runner.o

runner.o: runner.cc
	$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc

clean:
	rm -f runner.o
