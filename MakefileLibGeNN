# Include common makefile
include MakefileCommon

# Find source files
FRONTEND_SOURCES	:=$(wildcard src/genn/*.cc)
BACKEND_SOURCES		:=$(wildcard src/genn/code_generator/*.cc)
SOURCES			:=$(FRONTEND_SOURCES) $(BACKEND_SOURCES) 

# Remove src prefix and replace with object directory
OBJECTS			:=$(SOURCES:src/%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all

all: $(LIBGENN)

$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: src/%.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;
