# Include common makefile
include ../../genn/MakefileCommon

# Add SpineML commoninclude directory to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/spineml/common

# Find source files
PUGIXML_SOURCES		:=$(wildcard pugixml/*.cc)
SOURCES			:=$(wildcard *.cc) $(PUGIXML_SOURCES)

# Add prefix to object directory and library name
SPINEML_COMMON		:=$(GENN_DIR)/lib/libspineml_common$(GENN_PREFIX).a

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/spineml/common

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean

all: $(SPINEML_COMMON)

$(SPINEML_COMMON): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++11 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@find $(OBJECT_DIRECTORY) -type f -name "*.o" -delete
	@find $(OBJECT_DIRECTORY) -type f -name "*.d" -delete
	@rm -f $(SPINEML_COMMON)
