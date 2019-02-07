# Include common makefile
include ../MakefileCommon

# If CUDA path isn't set, default to standard path for (at least Ubuntu) Linux systems
# **NOTE** setting CUDA_PATH is a REQUIRED post-installation action when installing CUDA so this shouldn't be required
CUDA_PATH		?=/usr/local/cuda

# Add backend and CUDA include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn_cuda_backend -I"$(CUDA_PATH)/include"

# Add prefix to object directory and library name
BACKEND			:=$(GENN_DIR)/lib/libgenn_cuda_backend$(GENN_PREFIX).a

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn_cuda_backend

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

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(BACKEND)
