# Include common makefile
include ../../MakefileCommon

# Add DVS and LIBCAER include directories to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn/sensors/dvs $(shell pkg-config libcaer --cflags)

# Add prefix to object directory and library name
LIBDVS	:=$(LIBRARY_DIRECTORY)/libgenn_dvs$(GENN_PREFIX).$(LIBRARY_EXTENSION)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/sensors/dvs

# Find source files
SOURCES			:= $(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all

all: $(LIBDVS)

ifdef DYNAMIC
ifeq ($(DARWIN),DARWIN)
$(LIBDVS): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -dynamiclib -undefined dynamic_lookup $(CXXFLAGS) -o $@ $(OBJECTS)
	install_name_tool -id "@loader_path/$(@F)" $@
else
$(LIBDVS): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(OBJECTS)
endif
else
$(LIBDVS): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)
endif

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(LIBDVS)
