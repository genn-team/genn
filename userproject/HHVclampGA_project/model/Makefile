GENERATED_CODE_DIR	:=HHVClamp_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: HHVClamp

HHVClamp: VClampGA.cc generated_code
	$(CXX) $(CXXFLAGS)  -I../../include VClampGA.cc -o HHVClamp -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
