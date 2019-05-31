# Include common makefile
include ../../genn/MakefileCommon

# Add prefix to object directory and library name
SPINEML_SIMULATOR_STANDALONE	:=$(GENN_DIR)/bin/spineml_simulator$(GENN_PREFIX)
OBJECT_DIRECTORY		:=$(OBJECT_DIRECTORY)/spineml/simulator_standalone

# Find source files
SOURCES			:=$(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Add compiler and linker flags to link libGeNN and correct backend; and to configure generator to use backend
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml/common -I$(GENN_DIR)/include/spineml/simulator
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_simulator$(GENN_PREFIX) -lspineml_common$(GENN_PREFIX) -ldl

.PHONY: all clean spineml_common spineml_simulator

all: $(SPINEML_SIMULATOR_STANDALONE)

$(SPINEML_SIMULATOR_STANDALONE): $(OBJECTS) spineml_common spineml_simulator
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++11 $(CXXFLAGS) -c -o $@ $<

%.d: ;

spineml_common:
	$(MAKE) -C $(GENN_DIR)/src/spineml/common

spineml_simulator:
	$(MAKE) -C $(GENN_DIR)/src/spineml/simulator

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(SPINEML_SIMULATOR_STANDALONE)
