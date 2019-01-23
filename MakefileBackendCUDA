# Include common makefile
include ../../MakefileCommon

# Check CUDA is present
ifndef CUDA_PATH
	$(error Environment variable CUDA_PATH must be defined)
endif
	
# Add include paths to CXX flags
CXXFLAGS		+=-Iinclude -I../../lib_genn/include -I../../third_party -I"$(CUDA_PATH)/include"

# Add prefix to object directory and library name
OBJECT_DIRECTORY	:=obj$(GENN_PREFIX)
BACKEND			:=$(GENN_DIR)/lib/libcuda_backend$(GENN_PREFIX).a

# Find source files
SOURCES			:= $(wildcard src/*.cc)

# Remove src prefix and replace with object directory
OBJECTS			:=$(SOURCES:src/%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean

all: $(BACKEND)

$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: src/%.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECTS) $(DEPS) $(BACKEND)
