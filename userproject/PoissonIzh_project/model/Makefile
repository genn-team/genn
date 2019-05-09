GENERATED_CODE_DIR	:=PoissonIzh_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: PoissonIzh

PoissonIzh: PoissonIzhSim.cc generated_code
	$(CXX) $(CXXFLAGS)  -I../../include PoissonIzhSim.cc -o PoissonIzh -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
