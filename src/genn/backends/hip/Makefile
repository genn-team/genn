# Include common makefile
include ../../MakefileCommon

# Check that CUDA path is set
ifndef HIP_PATH
    $(error Environment variable HIP_PATH must be defined)
endif


# Add backend and CUDA include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn/backends/hip -I"$(HIP_PATH)/include" -pthread

# Add prefix to object directory and library name
BACKEND			:=$(LIBRARY_DIRECTORY)/libgenn_hip_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/backends/hip

# Find source files
SOURCES			:= $(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Select platform
ifeq ($(HIP_PLATFORM), nvidia)
	CXXFLAGS+=-D__HIP_PLATFORM_NVIDIA__
        CXXFLAGS+=-I$(CUDA_PATH)/include
else
	CXXFLAGS+=-D__HIP_PLATFORM_AMD__
endif

.PHONY: all

all: $(BACKEND)

ifdef DYNAMIC
$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(OBJECTS)
else
$(BACKEND): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)
endif

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(BACKEND)
