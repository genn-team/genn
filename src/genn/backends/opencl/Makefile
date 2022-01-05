# Include common makefile
include ../../MakefileCommon

# Check that OpenCL path is set
# **NOTE** on MacOS X we can just use framework
ifneq ($(DARWIN),DARWIN)
    ifndef OPENCL_PATH
        $(error Environment variable OPENCL_PATH must be defined - this is a required post-installation action when installing OpenCL)
    endif
endif

# Add backend and OpenCL include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn/backends/opencl -Wno-ignored-attributes -DCL_HPP_TARGET_OPENCL_VERSION=120 -DCL_HPP_MINIMUM_OPENCL_VERSION=120
ifneq ($(DARWIN),DARWIN)
    CXXFLAGS		+=-I$(OPENCL_PATH)/include
endif

# Add prefix to object directory and library name
BACKEND			:=$(LIBRARY_DIRECTORY)/libgenn_opencl_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/backends/opencl

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
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(BACKEND)
