# Include common makefile
include ../MakefileCommon

# Find source files
FRONTEND_SOURCES	:=$(wildcard *.cc)
BACKEND_SOURCES		:=$(wildcard code_generator/*.cc)
SOURCES			:=$(FRONTEND_SOURCES) $(BACKEND_SOURCES) 

# Build objecs in sub-directory
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/genn

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean

all: $(LIBGENN)

$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@find $(OBJECT_DIRECTORY) -type f -name "*.o" -delete
	@find $(OBJECT_DIRECTORY) -type f -name "*.d" -delete
	@rm -f $(LIBGENN)
