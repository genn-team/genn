# Include common makefile
include MakefileCommon

# Find source files
FRONTEND_SOURCES	:=$(wildcard *.cc)
BACKEND_SOURCES		:=$(wildcard code_generator/*.cc)
SOURCES			:=$(FRONTEND_SOURCES) $(BACKEND_SOURCES) 

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all

all: $(LIBGENN)

$(LIBGENN): $(OBJECTS)
	mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;
