# Include common makefile
include ../MakefileCommon

# Add compiler and linker flags to link libGeNN and correct backend; and to configure generator to use backend
LDFLAGS			+= -L$(LIBRARY_DIRECTORY) -lgenn_fenn_backend$(GENN_PREFIX) -lgenn$(GENN_PREFIX) -ldl $(shell pkg-config libffi --libs)
CXXFLAGS		+= -I$(GENN_DIR)/include/genn/backends/fenn

# Determine full path to generator and backend
FENN_TEST		:=$(GENN_DIR)/bin/fenn_test$(GENN_PREFIX)

# Find source files
SOURCES			:= $(wildcard *.cc)

# Add object directory prefix
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

.PHONY: all clean libgenn fenn_backend

all: $(FENN_TEST)


$(FENN_TEST): $(OBJECTS) libgenn fenn_backend
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

generator.d: ;

libgenn:
	if [ -w $(LIBRARY_DIRECTORY) ]; then $(MAKE) -C $(GENN_DIR)/src/genn/genn; fi;

fenn_backend:
	if [ -w $(LIBRARY_DIRECTORY) ]; then $(MAKE) -C $(GENN_DIR)/src/genn/backends/fenn; fi;

clean:
	rm -f generator.d $(GENERATOR)
