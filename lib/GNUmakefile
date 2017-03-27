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


# generateALL and libgenn.a targets
ifndef CPU_ONLY
    GENERATEALL          :=$(GENERATEALL_PATH)/generateALL
else
    GENERATEALL          :=$(GENERATEALL_PATH)/generateALL_CPU_ONLY
endif

# Target rules
.PHONY: all clean clean_libgenn clean_generateall always_compile

all: $(GENERATEALL)

$(GENERATEALL): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -DMODEL=\"$(MODEL)\" -o $@ $(SRC_PATH)/generate*.cc $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean: clean_generateall clean_libgenn

clean_generateall:
	rm -f $(GENERATEALL) $(GENERATEALL).d

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))