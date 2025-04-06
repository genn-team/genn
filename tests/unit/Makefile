# Include common makefile
include ../../src/genn/MakefileCommon

UNIT_TEST_PATH      := $(GENN_DIR)/tests/unit
OBJECT_DIRECTORY    := $(OBJECT_DIRECTORY)/tests/unit
GTEST_BUILD_DIR     := $(GTEST_DIR)/build

SOURCES             := $(wildcard *.cc)
OBJECTS             := $(SOURCES:%.cc=$(OBJECT_DIRECTORY)/%.o)
DEPS                := $(OBJECTS:.o=.d)

# Link against compiled GTest static libraries
LDFLAGS             += -L$(LIBRARY_DIRECTORY) -lgenn_single_threaded_cpu_backend$(GENN_PREFIX) -lgenn$(GENN_PREFIX) \
                       -lpthread -ldl $(shell pkg-config libffi --libs) \
                       -L$(GTEST_BUILD_DIR) -lgtest -lgtest_main

CXXFLAGS            += -I$(GENN_DIR)/include/genn/backends/single_threaded_cpu \
                       -I"$(GTEST_DIR)/include" -std=c++17

TEST                := $(UNIT_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean libgenn backend

all: $(TEST)

$(TEST): $(OBJECTS) libgenn backend
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

-include $(DEPS)

$(OBJECT_DIRECTORY)/%.o: %.cc $(OBJECT_DIRECTORY)/%.d
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

libgenn:
	if [ -w $(GENN_DIR)/lib ]; then $(MAKE) -C $(GENN_DIR)/src/genn/genn; fi;

backend:
	if [ -w $(GENN_DIR)/lib ]; then $(MAKE) -C $(GENN_DIR)/src/genn/backends/single_threaded_cpu; fi;

clean:
	rm -f $(OBJECT_DIRECTORY)/*.o $(OBJECT_DIRECTORY)/*.d $(TEST)
