# Include common makefile
include ../MakefileCommon

# Add SpineML commoninclude directory to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/spineml_common

# Find source files
PUGIXML_SOURCES		:=$(wildcard pugixml/*.cpp)
SOURCES			:=pugixml/pugixml.cpp#$(PUGIXML_SOURCES)

# Add prefix to object directory and library name
SPINEML_COMMON		:=$(GENN_DIR)/lib/libspineml_common$(GENN_PREFIX).a

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/spineml_common

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cpp=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean

all: $(SPINEML_COMMON)

$(SPINEML_COMMON): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cpp $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@find $(OBJECT_DIRECTORY) -type f -name "*.o" -delete
	@find $(OBJECT_DIRECTORY) -type f -name "*.d" -delete
	@rm -f $(LIBGENN)
