# Include common makefile
include ../genn/MakefileCommon

# Add backend  include directory to compiler flags
CXXFLAGS		+=-I$(GENN_DIR)/include/genn_single_threaded_cpu_backend

# Add prefix to object directory and library name
BACKEND			:=$(GENN_DIR)/lib/libgenn_single_threaded_cpu_backend$(GENN_PREFIX).a

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
