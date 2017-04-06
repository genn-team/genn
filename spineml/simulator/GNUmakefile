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


# Makefile for the GeNN "generateSpineML" executable
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------
# Because we're including another Makefile which includes 
# its own goals, we need to manually specify the DEFAULT_GOAL
.DEFAULT_GOAL := all

# Include makefile which builds libgenn
include $(GENN_PATH)/lib/GNUMakefileLibGeNN

# Get simulate SpineML path i.e. directory of this Makefile
SIMULATE_SPINEML_PATH         :=$(CURDIR)

# Get SpineML path i.e. parent directory
SPINEML_PATH                  :=$(abspath $(SIMULATE_SPINEML_PATH)/..)

COMMON_SPINEML_PATH           :=$(SPINEML_PATH)/common

# generateALL and libgenn.a targets
ifndef CPU_ONLY
    SIMULATE_SPINEML          =$(SIMULATE_SPINEML_PATH)/simulateSpineML
else
    SIMULATE_SPINEML          =$(SIMULATE_SPINEML_PATH)/simulateSpineML_CPU_ONLY
endif

SIMULATE_SPINEML_SOURCES      := common/pugixml/pugixml.cpp simulator/main.cc
SIMULATE_SPINEML_SOURCES      :=$(addprefix $(SPINEML_PATH)/,$(SIMULATE_SPINEML_SOURCES))

INCLUDE_FLAGS                 += -I $(SIMULATE_SPINEML_PATH) -I $(COMMON_SPINEML_PATH)
LINK_FLAGS                    += -ldl

# Target rules
.PHONY: all clean clean_simulate_spine_ml always_compile

all: $(SIMULATE_SPINEML)

$(SIMULATE_SPINEML): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -o $@ $(SIMULATE_SPINEML_SOURCES) $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_generate_spine_ml clean_libgenn

clean_simulate_spine_ml:
	rm -f $(SIMULATE_SPINEML) $(SIMULATE_SPINEML).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))