GENERATED_CODE_DIR	:=Schmuker_2014_classifier_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra -Wno-reorder -Wno-sign-compare -Wno-unused-variable -Wno-unused-parameter

.PHONY: all clean generated_code

all: Schmuker_2014_classifier

Schmuker_2014_classifier: experiment.cc generated_code
	$(CXX) $(CXXFLAGS)  experiment.cc -o Schmuker_2014_classifier -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
