# Include common makefile
include ../MakefileCommon

# Add standard include paths to CXX flags
CXXFLAGS 		+=-Iinclude -I../third_party

# Get directory of this makefile i.e. libgenn (means make can be invoked else where)
LIBGENN_DIR 		:= $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Add prefix to object directory and library name
OBJECT_DIRECTORY	:=obj$(GENN_PREFIX)
LIBGENN			:=lib/libgenn$(GENN_PREFIX).a

# Find source files
FRONTEND_SOURCES	:=$(wildcard src/*.cc)
BACKEND_SOURCES		:=$(wildcard src/code_generator/*.cc)
SOURCES			:=$(FRONTEND_SOURCES) $(BACKEND_SOURCES) 

# Remove src prefix and replace with 
OBJECTS			:=$(SOURCES:src/%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean

all: $(LIBGENN)

$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: src/%.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECTS) $(DEPS) $(LIBGENN)
