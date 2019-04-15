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
SPINEML_SIM_SRC_PATH	:= $(GENN_DIR)/src/spineml/simulator

TEST_SOURCES		:= $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc
TEST_SOURCES		+= $(SPINEML_SIM_TEST_PATH)/*.cc
TEST_SOURCES		+= $(SPINEML_SIM_SRC_PATH)/connectors.cc $(SPINEML_SIM_SRC_PATH)/inputValue.cc $(SPINEML_SIM_SRC_PATH)/networkClient.cc

# Add compiler and linker flags to link libGeNN and pthreads
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_common$(GENN_PREFIX) -lpthread
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml/common -I$(GENN_DIR)/include/spineml/simulator -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" 

# Determine full path to generator and backend
TEST			:=$(SPINEML_SIM_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean libgenn

all: $(TEST)

$(TEST): $(TEST_SOURCES) spineml_common
	$(CXX) $(CXXFLAGS) $(TEST_SOURCES) -o $@ $(LDFLAGS)

spineml_common:
	$(MAKE) -C $(GENN_DIR)/src/spineml/common

clean:
	rm -f $(TEST) *.d *.gcno
