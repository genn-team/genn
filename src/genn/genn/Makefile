# Include common makefile
include ../MakefileCommon

# Find source files
FRONTEND_SOURCES	:=$(wildcard *.cc)
CODE_GENERATOR_SOURCES	:=$(wildcard code_generator/*.cc)
TRANSPILER_SOURCES	:=$(wildcard transpiler/*.cc)
RUNTIME_SOURCES		:=$(wildcard runtime/*.cc)
SOURCES			:=$(FRONTEND_SOURCES) $(CODE_GENERATOR_SOURCES) $(TRANSPILER_SOURCES) $(RUNTIME_SOURCES)

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn/genn

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Read GeNN version and git hash
GENN_VERSION		:= $(shell cat "$(GENN_DIR)/version.txt")
GIT_HASH		:= $(shell git -C "$(GENN_DIR)" rev-parse HEAD 2>/dev/null)

# Add version and hash to compiler flags
CXXFLAGS		+= -DGENN_VERSION=\"$(GENN_VERSION)\" -DGIT_HASH=\"$(GIT_HASH)\"

.PHONY: all clean

all: $(LIBGENN)

ifdef DYNAMIC
ifeq ($(DARWIN),DARWIN)
$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -dynamiclib -undefined dynamic_lookup $(CXXFLAGS) -o $@ $(OBJECTS)
	install_name_tool -id "@loader_path/$(@F)" $@
else
$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(OBJECTS)
endif
else
$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)
endif

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@find $(OBJECT_DIRECTORY) -type f -name "*.o" -delete
	@find $(OBJECT_DIRECTORY) -type f -name "*.d" -delete
	@rm -f $(LIBGENN)
