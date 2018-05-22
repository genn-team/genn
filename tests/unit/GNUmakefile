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


# Makefile for the GeNN simulator test executable
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------
# Because we're including another Makefile which includes 
# its own goals, we need to manually specify the DEFAULT_GOAL
.DEFAULT_GOAL := all

# Include makefile which builds libgenn
include $(GENN_PATH)/lib/GNUMakefileLibGeNN

# Get simulate SpineML path i.e. directory of this Makefile
TEST_PATH         :=$(CURDIR)

# generateALL and libgenn.a targets
TEST          =$(TEST_PATH)/test

GTEST_SOURCES     := $(GTEST_DIR)/src/gtest-all.cc

TEST_SOURCES      := $(GTEST_SOURCES) $(TEST_PATH)/*.cc
INCLUDE_FLAGS                 += -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" -I $(SPINEML_PATH)/simulator -I $(SPINEML_PATH)/common
LINK_FLAGS                    += -lpthread

# Target rules
.PHONY: all clean clean_test always_compile

all: $(TEST)

$(TEST): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SOURCES) $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_test clean_libgenn

clean_test:
	rm -f $(TEST) $(TEST).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))