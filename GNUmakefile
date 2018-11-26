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

# Get generate SpineML path i.e. directory of this Makefile
NUGENN_PATH         :=$(CURDIR)

# generateALL and libgenn.a targets
ifndef CPU_ONLY
    NUGENN                   =$(NUGENN_PATH)/nugenn
else
    NUGENN                   =$(NUGENN_PATH)/nugenn_CPU_ONLY
endif

NUGENN_SOURCES      := cpu_code_generator.cc  cuda_code_generator.cc  main.cc
NUGENN_SOURCES      :=$(addprefix $(NUGENN_PATH)/,$(NUGENN_SOURCES))

# Target rules
.PHONY: all clean clean_generate_nugenn always_compile

all: $(NUGENN)

$(NUGENN): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -o $@ $(NUGENN_SOURCES) -DGENERATOR_MAIN_HANDLED $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_generate_nugenn clean_libgenn

clean_generate_nugenn:
	rm -f $(NUGENN) $(NUGENN).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))
