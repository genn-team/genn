# Include common makefile
include ../../src/genn/MakefileCommon

# Get simulate SpineML path i.e. directory of this Makefile
UNIT_TEST_PATH		:=$(GENN_DIR)/tests/unit
OBJECT_DIRECTORY	:=$(OBJECT_DIRECTORY)/tests/unit

SOURCES			:=$(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc $(wildcard *.cc)
OBJECTS			:=$(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS			:=$(OBJECTS:.o=.d)

# Add compiler and linker flags to link libGeNN and pthreads
LDFLAGS			+= -L$(LIBRARY_DIRECTORY) -lgenn_single_threaded_cpu_backend$(GENN_PREFIX) -lgenn$(GENN_PREFIX) -lpthread
CXXFLAGS		+= -I$(GENN_DIR)/include/genn/backends/single_threaded_cpu -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" 

# Determine full path to generator and backend
TEST			:=$(UNIT_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean libgenn backend

all: $(TEST)

$(TEST): $(OBJECTS) libgenn backend
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) -std=c++17 $(CXXFLAGS) -c -o $@ $<

%.d: ;

libgenn:
	if [ -w $(GENN_DIR)/lib ]; then $(MAKE) -C $(GENN_DIR)/src/genn/genn; fi;

backend:
	if [ -w $(GENN_DIR)/lib ]; then $(MAKE) -C $(GENN_DIR)/src/genn/backends/single_threaded_cpu; fi;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(TEST)
