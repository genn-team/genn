# Include common makefile
include ../MakefileCommon

# Determine full path to simulator
SPINEML_SIMULATOR	:=$(GENN_DIR)/bin/spineml_simulator$(GENN_PREFIX)
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/spineml_simulator

# Find source files
SOURCES			:=$(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Add compiler and linker flags to link libGeNN and correct backend; and to configure generator to use backend
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml_common -I$(GENN_DIR)/include/spineml_simulator
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_common$(GENN_PREFIX) -ldl

.PHONY: all clean spineml_common

all: $(SPINEML_SIMULATOR)

$(SPINEML_SIMULATOR): $(OBJECTS) spineml_common
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

spineml_common:
	$(MAKE) -C $(GENN_DIR)/src/spineml_common

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(SPINEML_SIMULATOR)
