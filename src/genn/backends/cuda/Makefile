# Include common makefile
include ../../MakefileCommon

# Check that CUDA path is set
ifndef CUDA_PATH
    $(error Environment variable CUDA_PATH must be defined - this is a required post-installation action when installing CUDA)
endif

# Add backend and CUDA include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn/backends/cuda -I"$(CUDA_PATH)/include" -pthread

# Add prefix to object directory and library name
BACKEND			:=$(LIBRARY_DIRECTORY)/libgenn_cuda_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/backends/cuda

# Find source files
SOURCES			:= $(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all

all: $(BACKEND)

ifdef DYNAMIC
ifeq ($(DARWIN),DARWIN)
$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -dynamiclib -undefined dynamic_lookup $(CXXFLAGS) -o $@ $(OBJECTS)
	install_name_tool -id "@loader_path/$(@F)" $@
else
$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(OBJECTS)
endif
else
$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)
endif

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++11 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(BACKEND)
