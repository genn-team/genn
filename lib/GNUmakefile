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


# Makefile for the GeNN "generateALL" executable
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------
# Because we're including another Makefile which includes 
# its own goals, we need to manually specify the DEFAULT_GOAL
.DEFAULT_GOAL := all

# Include makefile which builds libgenn
include $(GENN_PATH)/lib/GNUMakefileLibGeNN

# Source, Include, Object and Library directories
GENERATEALL_PATH         :=$(CURDIR)

# BUILD_MODEL_INCLUDE contains a colon-seperated list of additional include paths.
# substitute :s for spaces and then prepend each path with -I so it gets turned into an include directory
INCLUDE_FLAGS            +=$(patsubst %,-I%,$(subst :, ,$(BUILD_MODEL_INCLUDE)))


# generateALL target
GENERATEALL          :=$(GENERATEALL_PATH)/generateALL$(LIBGENN_PREFIX)

# Target rules
.PHONY: all clean clean_libgenn clean_generateall always_compile pygenn clean_pygenn

all: $(GENERATEALL)

$(GENERATEALL): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -DMODEL=\"$(MODEL)\" -o $@ $(SRC_PATH)/generate*.cc $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_generateall clean_libgenn

clean_generateall:
	rm -f $(GENERATEALL) $(GENERATEALL).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))
