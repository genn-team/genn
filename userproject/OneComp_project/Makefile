CXXFLAGS+=-std=c++11 -I../include

.PHONY: all clean

all: generate_run

generate_run: generate_run.cc
	$(CXX) generate_run.cc $(CXXFLAGS) -o generate_run

clean:
	rm -f generate_run
