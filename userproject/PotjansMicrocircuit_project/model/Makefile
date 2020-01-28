GENERATED_CODE_DIR	:=PotjansMicrocircuit_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: PotjansMicrocircuit

PotjansMicrocircuit: PotjansMicrocircuitSim.cc generated_code
	$(CXX) $(CXXFLAGS) -I../../include PotjansMicrocircuitSim.cc -ldl -o PotjansMicrocircuit

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
