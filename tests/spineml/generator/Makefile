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
SPINEML_GEN_TEST_PATH	:=$(GENN_DIR)/tests/spineml/generator

TEST_SOURCES		:= $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc
TEST_SOURCES		+= $(SPINEML_GEN_TEST_PATH)/*.cc
TEST_SOURCES		+= $(GENN_DIR)/src/spineml/generator/modelCommon.cc

# Add compiler and linker flags to link SpineML simulator and common libraries; and pthreads
LDFLAGS			+= -L$(GENN_DIR)/lib -lspineml_common$(GENN_PREFIX) -lgenn$(GENN_PREFIX) -lpthread
CXXFLAGS		+= -I$(GENN_DIR)/include/spineml/common -I$(GENN_DIR)/include/spineml/generator -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" 

# Determine full path to test executable
TEST			:=$(SPINEML_GEN_TEST_PATH)/test$(GENN_PREFIX)

.PHONY: all clean genn spineml_common

all: $(TEST)

$(TEST): $(TEST_SOURCES) genn spineml_common
	$(CXX) -std=c++11 $(CXXFLAGS) $(TEST_SOURCES) -o $@ $(LDFLAGS)

genn:
	$(MAKE) -C $(GENN_DIR)/src/genn/genn

spineml_common:
	$(MAKE) -C $(GENN_DIR)/src/spineml/common

clean:
	rm -f $(TEST) *.d *.gcno
