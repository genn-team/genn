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


# Source, Include, Object and Library directories
GENERATE_SPINEML_PATH         :=$(CURDIR)


# generateALL and libgenn.a targets
ifndef CPU_ONLY
    GENERATE_SPINEML          =$(GENERATE_SPINEML_PATH)/generateSpineML
else
    GENERATE_SPINEML          =$(GENERATE_SPINEML_PATH)/generateSpineML_CPU_ONLY
endif

GENERATE_SPINEML_SOURCES      := pugixml/pugixml.cpp main.cc spineMLNeuronModel.cc
GENERATE_SPINEML_SOURCES      :=$(addprefix $(GENERATE_SPINEML_PATH)/,$(GENERATE_SPINEML_SOURCES))

INCLUDE_FLAGS                 += -I $(GENERATE_SPINEML_PATH)/pugixml

# Target rules
.PHONY: all clean clean_generate_spine_ml always_compile

all: $(GENERATE_SPINEML)

$(GENERATE_SPINEML): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_PATH)/generate*.cc $(GENERATE_SPINEML_SOURCES) -DGENERATOR_MAIN_HANDLED $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_generate_spine_ml clean_libgenn

clean_generate_spine_ml:
	rm -f $(GENERATE_SPINEML) $(GENERATE_SPINEML).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))