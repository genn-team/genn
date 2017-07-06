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
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------
# Because we're including another Makefile which includes 
# its own goals, we need to manually specify the DEFAULT_GOAL
.DEFAULT_GOAL := all

# These unit tests
CPU_ONLY = 1

# Include makefile which builds libgenn
include $(GENN_PATH)/lib/GNUMakefileLibGeNN

# Get simulate SpineML path i.e. directory of this Makefile
TEST_PATH         :=$(CURDIR)

# Get SpineML path
SPINEML_PATH                  :=$(GENN_PATH)/spineml

# generateALL and libgenn.a targets
TEST          =$(TEST_PATH)/test


SPINEML_SOURCES   := $(SPINEML_PATH)/common/pugixml/pugixml.cpp $(SPINEML_PATH)/simulator/connectors.cc
GTEST_SOURCES     := $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc

TEST_SOURCES      := $(SPINEML_SOURCES) $(GTEST_SOURCES) $(TEST_PATH)/*.cc

INCLUDE_FLAGS                 += -I $(SPINEML_PATH)/simulator -I $(SPINEML_PATH)/common
LINK_FLAGS                    += -lpthreads

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

CXX = g++
CXXFLAGS = -std=c++11 -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" -I "$(GENN_PATH)/lib/include" -I "$(GENN_PATH)/spineml/simulator" -I "$(GENN_PATH)/spineml/common" -DCPU_ONLY
LINK_FLAGS = -lpthread

SOURCES = connectors.cc $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc $(GENN_PATH)/lib/src/codeGenUtils.cc
OBJECTS =$(foreach obj,$(basename $(SOURCES)),$(obj).o)

%.o: %.cc
	$(CXX)  -c -o $@ $< $(CXXFLAGS)

test: $(OBJECTS) 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LINK_FLAGS)
	
clean:
	rm -f *.o test