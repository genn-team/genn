# Include common makefile
include ../genn/MakefileCommon

# Check CUDA is present
ifndef CUDA_PATH
	$(error Environment variable CUDA_PATH must be defined)
endif

# Add backend and CUDA include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn_cuda_backend -I"$(CUDA_PATH)/include"

# Add prefix to object directory and library name
BACKEND			:=$(GENN_DIR)/lib/libgenn_cuda_backend$(GENN_PREFIX).a

# Find source files
SOURCES			:= $(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all

all: $(BACKEND)

$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;