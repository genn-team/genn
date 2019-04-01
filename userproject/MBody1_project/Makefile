CXXFLAGS+=-std=c++11 -I../include -MMD -MP

.PHONY: all clean

all: generate_run

-include generate_run.d

generate_run: generate_run.cc generate_run.d
	$(CXX) generate_run.cc $(CXXFLAGS) -o generate_run

%.d: ;

clean:
	rm -f generate_run generate_run.d
