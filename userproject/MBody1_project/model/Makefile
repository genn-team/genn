GENERATED_CODE_DIR	:=MBody1_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: MBody1

MBody1: MBody1Sim.cc generated_code
	$(CXX) $(CXXFLAGS)  -I../../include/ MBody1Sim.cc -o MBody1 -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
