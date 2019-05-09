GENERATED_CODE_DIR	:=OneComp_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: OneComp

OneComp: OneCompSim.cc generated_code
	$(CXX) $(CXXFLAGS)  -I../../include OneCompSim.cc -o OneComp -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
