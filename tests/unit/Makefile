# Include common makefile
include ../../src/genn/MakefileCommon

# Get simulate SpineML path i.e. directory of this Makefile
UNIT_TEST_PATH		:=$(GENN_DIR)/tests/unit

TEST_SOURCES		:= $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc $(UNIT_TEST_PATH)/*.cc

# Add compiler and linker flags to link libGeNN and pthreads
LDFLAGS			+= -L$(GENN_DIR)/lib -lgenn$(GENN_PREFIX) -lpthread
CXXFLAGS		+= -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" 

# Determine full path to generator and backend
TEST			:=$(UNIT_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean libgenn

all: $(TEST)

$(TEST): $(TEST_SOURCES) libgenn
	$(CXX) -std=c++11 $(CXXFLAGS) $(TEST_SOURCES) -o $@ $(LDFLAGS)

generator.d: ;

libgenn:
	if [ -w $(GENN_DIR)/lib ]; then $(MAKE) -C $(GENN_DIR)/src/genn/genn; fi;

clean:
	rm -f $(TEST) *.d *.gcno
