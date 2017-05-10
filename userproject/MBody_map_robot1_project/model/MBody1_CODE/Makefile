
CXXFLAGS       :=-c -DCPU_ONLY 

INCLUDEFLAGS   =-I"$(GENN_PATH)/lib/include"

all: runner.o

runner.o: runner.cc
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc

clean:
	rm -f runner.o
