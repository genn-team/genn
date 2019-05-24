##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##              Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------
# Makefile for the SpineML simulator test executable
# Include common makefile
include ../../../src/genn/MakefileCommon

# Get simulate SpineML path i.e. directory of this Makefile
SPINEML_SIM_TEST_PATH	:=$(GENN_DIR)/tests/spineml/simulator

TEST_SOURCES		:= $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc
TEST_SOURCES		+= $(SPINEML_SIM_TEST_PATH)/*.cc

# Add compiler and linker flags to link SpineML simulator and common libraries; and pthreads
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_simulator$(GENN_PREFIX) -lspineml_common$(GENN_PREFIX) -lpthread
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml/common -I$(GENN_DIR)/include/spineml/simulator -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" 

# Determine full path to test executable
TEST			:=$(SPINEML_SIM_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean spineml_common spineml_simulator

all: $(TEST)

$(TEST): $(TEST_SOURCES) spineml_common spineml_simulator
	$(CXX) -std=c++11 $(CXXFLAGS) $(TEST_SOURCES) -o $@ $(LDFLAGS)

spineml_common:
	$(MAKE) -C $(GENN_DIR)/src/spineml/common

spineml_simulator:
	$(MAKE) -C $(GENN_DIR)/src/spineml/simulator
clean:
	rm -f $(TEST) *.d *.gcno
