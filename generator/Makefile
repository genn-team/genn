CXXFLAGS += -std=c++11 -Wall -Wpedantic -Wextra  -I../backends/single_threaded_cpu/include -I../lib_genn/include -I../third_party -DMODEL=\"test_model.cc\" -DBACKEND_TYPE=SingleThreadedCPU -I$(BOB_ROBOTICS_PATH) -I ~/genn_examples

.PHONY: all clean libgenn backend

all: generator

generator: generator.o libgenn backend
	$(CXX) $(CXXFLAGS) generator.cc -o generator -L../lib_genn -L../backends/single_threaded_cpu -lgenn -lsingle_threaded_cpu_backend

libgenn:
	$(MAKE) -C ../lib_genn

backend:
	$(MAKE) -C ../backends/single_threaded_cpu
	
generator.o: generator.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f generator.o generator
