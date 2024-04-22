# Include common makefile
include ../../MakefileCommon

# Add backend  include directory to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn/backends/single_threaded_cpu

# Add prefix to object directory and library name
BACKEND			:=$(LIBRARY_DIRECTORY)/libgenn_single_threaded_cpu_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/backends/single_threaded_cpu

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
	@rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(BACKEND)
