CXXFLAGS+=-std=c++11 -I../include -MMD -MP

.PHONY: all clean gen_input_structured

all: generate_run

-include generate_run.d

generate_run: generate_run.cc generate_run.d gen_input_structured
	$(CXX) generate_run.cc $(CXXFLAGS) -o generate_run

%.d: ;

gen_input_structured:
	$(MAKE) -C ../tools gen_input_structured

clean:
	rm -f generate_run generate_run.d
