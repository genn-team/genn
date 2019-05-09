GENERATED_CODE_DIR	:=SynDelay_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: SynDelay

SynDelay: SynDelaySim.cc generated_code
	$(CXX) $(CXXFLAGS)  -I../include SynDelaySim.cc -o SynDelay -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
