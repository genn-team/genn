# Include common makefile
include ../../genn/MakefileCommon

# Add prefix to object directory and library name
SPINEML_SIMULATOR	:=$(GENN_DIR)/lib/libspineml_simulator$(GENN_PREFIX).a
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/spineml/simulator

# Find source files
SOURCES			:=$(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Add compiler and linker flags to link libGeNN and correct backend; and to configure generator to use backend
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml/common -I$(GENN_DIR)/include/spineml/simulator
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_common$(GENN_PREFIX) -ldl

.PHONY: all clean

all: $(SPINEML_SIMULATOR)

$(SPINEML_SIMULATOR): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++11 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(SPINEML_SIMULATOR)
