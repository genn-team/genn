##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##           James Knight
##           Anton Komissarov
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##              Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2018-05-14
##
##--------------------------------------------------------------------------
# include a Makefile to build relocatable libgenn objects and define flags
.DEFAULT_GOAL := all
DYNAMIC   :=1
include $(GENN_PATH)/lib/GNUMakefileLibGeNN

# Define PyGeNN and SWIG related variables
SWIG                               :=swig -c++ -python
PYGENN_PATH                        :=$(GENN_PATH)/pygenn
PYGENN_OBJ_PATH                    :=$(PYGENN_PATH)/obj$(LIBGENN_PREFIX)
PYGENN_LIB_PATH                    :=$(PYGENN_PATH)/lib$(LIBGENN_PREFIX)
SWIG_PATH                          :=$(PYGENN_PATH)/swig
PYTHON_BIN                         ?=python
PYTHON_CONFIG                      :=$(PYTHON_BIN)-config
PYTHON_INCLUDE                     ?=$(shell $(PYTHON_CONFIG) --includes)
PYTHON_VERSION                     :=$(shell echo $(PYTHON_INCLUDE) | cut -f 1 -d " " | grep -e python... -o)
PYTHON_NUMPY_INCLUDE               ?=$(shell find $$($(PYTHON_CONFIG) --prefix)/lib -type d -path "*/*-packages/numpy/core/include" | grep -m 1 $(PYTHON_VERSION))

LIBGENN_PYGENN                    :=$(LIBGENN_PATH)/libgenn$(LIBGENN_PREFIX).so

# Check whether python includes are defined
ifeq ($(strip $(PYTHON_INCLUDE)),)
$(error attempt to automatically detect python include path failed. Try specifying the path as PYTHON_INCLUDE=)
endif
ifeq ($(strip $(PYTHON_NUMPY_INCLUDE)),)
$(error attempt to automatically detect python numpy package include path failed. Try specifying the path as PYTHON_NUMPY_INCLUDE=)
endif

INCLUDE_FLAGS +=-I$(SWIG_PATH)
INCLUDE_FLAGS +=$(PYTHON_INCLUDE)
INCLUDE_FLAGS +=-I$(PYTHON_NUMPY_INCLUDE)
INCLUDE_FLAGS +=-DGENERATOR_MAIN_HANDLED

GENERATOR_OBJ                      :=generateALL.o generateCPU.o generateInit.o generateKernels generateMPI.o generateRunner.o
GENERATOR_OBJ                      :=$(addprefix $(PYGENN_OBJ_PATH)/,$(GENERATOR_OBJ))

PYGENN_CUSTOM_OBJ                  :=newNeuronModelsCustom newPostsynapticModelsCustom newWeightUpdateModelsCustom currentSourceModelsCustom initVarSnippetCustom
PYGENN_CUSTOM_OBJ                  :=$(addprefix $(PYGENN_OBJ_PATH)/,$(PYGENN_CUSTOM_OBJ))
PYGENN_CUSTOM_OBJ                  :=$(addsuffix .o,$(PYGENN_CUSTOM_OBJ))

SWIG_SUBMODULES                    :=Snippet InitVarSnippet NewModels NeuronModels PostsynapticModels WeightUpdateModels CurrentSourceModels StlContainers SharedLibraryModel GeNNPreferences
SWIG_SUBMODULES_FQ                 :=$(addprefix $(PYGENN_PATH)/_,$(SWIG_SUBMODULES))
SWIG_SUBMODULES_FQ                 :=$(addsuffix .so,$(SWIG_SUBMODULES_FQ))

SWIG_CXX_FLAGS                     :=-shared -Wl,-rpath=$(LIBGENN_PATH)

.PHONY: all, clean, pygenn_pre, pygenn_post

# building pygenn is split in two parts:
# first generates the swig interfaces and custom models
# and the second builds and links
# Otherwise there are dependency issues when building in parallel (-j option)
all:
	$(MAKE) pygenn_pre
	$(MAKE) pygenn_post

# Auto-generate SWIG files
pygenn_pre: $(SWIG_PATH)
	python $(PYGENN_PATH)/generate_swig_interfaces.py $(GENN_PATH)

pygenn_post: $(PYGENN_OBJ_PATH), $(LIBGENN_PYGENN) $(SWIG_SUBMODULES_FQ) $(GENERATOR_OBJ)
	$(SWIG) -outdir $(SWIG_PATH) -o $(SWIG_PATH)/pygenn_wrap.cc $(SWIG_PATH)/pygenn.i
	$(CXX) $(CXXFLAGS) $(SWIG_CXX_FLAGS) $(SWIG_PATH)/pygenn_wrap.cc $(GENERATOR_OBJ) -o $(SWIG_PATH)/_pygenn.so $(INCLUDE_FLAGS) $(LINK_FLAGS)

# target to generate swig submodule shared libraries. SWIG expects that
# .so matches the module name and has a prefix "_".
$(PYGENN_PATH)/_%.so: $(SWIG_PATH)/%_wrap.cc $(LIBGENN_PYGENN)
	$(CXX) $(CXXFLAGS) $(SWIG_CXX_FLAGS) -o $@ $< $(INCLUDE_FLAGS) -ldl $(LINK_FLAGS)

# Generate wrapper C++ code from swig interfaces
$(SWIG_PATH)/%_wrap.cc: $(SWIG_PATH)/%.i
	$(SWIG) -outdir $(SWIG_PATH) -o $@ $<

# Build custom model objects
$(PYGENN_OBJ_PATH)/%.o: $(SWIG_PATH)/%.cc $(LIBGENN_OBJ_PATH)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I $(INC_PATH) -I $(SWIG_PATH)

# Build libgenn as shared library
$(LIBGENN_PYGENN): $(LIBGENN_PATH) $(PYGENN_CUSTOM_OBJ) $(LIBGENN_OBJ)
	$(CXX) -shared -o $@ $(LIBGENN_OBJ) $(PYGENN_CUSTOM_OBJ)

$(SWIG_PATH):
	mkdir -p $@

$(PYGENN_OBJ_PATH):
	mkdir -p $@

clean: clean_libgenn
	rm -rf $(SWIG_PATH)
	rm -rf $(PYGENN_OBJ_PATH)
