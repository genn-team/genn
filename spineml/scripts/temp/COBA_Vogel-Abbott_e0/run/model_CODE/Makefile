
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=--shared --compiler-options '-fPIC' -std=c++11 -x cu -arch sm_30 

INCLUDEFLAGS   =-I"$(GENN_PATH)/lib/include"

all: librunner.so

librunner.so: runner.cc
	$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc $(GENN_PATH)/lib/src/sparseUtils.cc -o librunner.so

clean:
	rm -f librunner.so
