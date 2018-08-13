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

# Define PyGeNN and SWIG related variables
SWIG                               :=swig -c++ -python
PYGENN                             :=pygenn
SWIG_PATH                          :=$(GENN_PATH)/lib/swig
PYGENN_PATH                        :=$(GENN_PATH)/lib/pygenn
PYGENN_SWIG_FILE_PATH              :=$(SWIG_PATH)/$(PYGENN).i
PYTHON_INCLUDE                     ?=$(shell python-config --includes)
# extract python version form PYTHON_INCLUDE
PYTHON_VERSION                     :=$(shell echo $(PYTHON_INCLUDE) | cut -f 1 -d " " | grep -e python... -o)
PYTHON_NUMPY_INCLUDE               ?=$(shell find $$(python-config --prefix)/lib -type d -path "*/site-packages/numpy/core/include" | grep -m 1 $(PYTHON_VERSION))

# include a Makefile to build libgenn objects and define flags
include $(GENN_PATH)/lib/GNUMakefileLibGeNN
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

GENERATOR_OBJ                      :=generateALL generateCPU generateInit generateKernels generateMPI generateRunner
GENERATOR_OBJ                      :=$(addprefix $(LIBGENN_OBJ_PATH)/,$(GENERATOR_OBJ))
GENERATOR_OBJ                      :=$(addsuffix .o,$(GENERATOR_OBJ))

PYGENN_CUSTOM_OBJ                  :=newNeuronModelsCustom newPostsynapticModelsCustom newWeightUpdateModelsCustom currentSourceModelsCustom initVarSnippetCustom
PYGENN_CUSTOM_OBJ                  :=$(addprefix $(LIBGENN_OBJ_PATH)/,$(PYGENN_CUSTOM_OBJ))
PYGENN_CUSTOM_OBJ                  :=$(addsuffix .o,$(PYGENN_CUSTOM_OBJ))

SWIG_SUBMODULES                    :=Snippet InitVarSnippet NewModels NeuronModels PostsynapticModels WeightUpdateModels CurrentSourceModels StlContainers SharedLibraryModel GeNNPreferences
SWIG_SUBMODULES_FQ                 :=$(addprefix $(PYGENN_PATH)/_,$(SWIG_SUBMODULES))
SWIG_SUBMODULES_FQ                 :=$(addsuffix .so,$(SWIG_SUBMODULES_FQ))

SWIG_CXX_FLAGS                     :=-shared -Wl,-rpath=$(LIBGENN_PATH)

# building pygenn is split in two parts:
# first generates the swig interfaces and custom models
# and the second builds and links
# Otherwise there are dependency issues when building in parallel (-j option)
pygenn:
	$(MAKE) -f GNUMakefilePyGeNN pygenn_pre
	$(MAKE) -f GNUMakefilePyGeNN pygenn_post

pygenn_pre: prepare_swig_config $(PYGENN_PATH)

pygenn_post: $(LIBGENN_PYGENN) $(SWIG_SUBMODULES_FQ) $(GENERATOR_OBJ)
	$(SWIG) $(SWIG_CPU) -outdir $(PYGENN_PATH) -o $(LIBGENN_OBJ_PATH)/$(PYGENN)_wrap.cc $(PYGENN_SWIG_FILE_PATH)
	$(CXX) $(CXXFLAGS) $(SWIG_CXX_FLAGS) $(LIBGENN_OBJ_PATH)/$(PYGENN)_wrap.cc \
           -o $(PYGENN_PATH)/_$(PYGENN).so $(GENERATOR_OBJ) $(INCLUDE_FLAGS) $(LINK_FLAGS)
	cp $(SWIG_PATH)/GeNNModel.py $(PYGENN_PATH)/GeNNModel.py
	cp $(SWIG_PATH)/GeNNGroups.py $(PYGENN_PATH)/GeNNGroups.py
	cp $(SWIG_PATH)/model_preprocessor.py $(PYGENN_PATH)/model_preprocessor.py

# target to generate swig submodule shared libraries. SWIG expects that
# .so matches the module name and has a prefix "_".
$(PYGENN_PATH)/_%.so: $(LIBGENN_OBJ_PATH)/%_wrap.cc $(LIBGENN_PYGENN)
	$(CXX) $(CXXFLAGS) $(SWIG_CXX_FLAGS) -o $@ $< $(INCLUDE_FLAGS) -ldl $(LINK_FLAGS)

# Generate wrapper C++ code from swig interfaces
$(LIBGENN_OBJ_PATH)/%_wrap.cc: $(SWIG_PATH)/%.i
	$(SWIG) $(SWIG_CPU) -outdir $(PYGENN_PATH) -o $@ $<

# Build custom model objects
$(LIBGENN_OBJ_PATH)/%.o: $(SWIG_PATH)/%.cc $(LIBGENN_OBJ_PATH)
	$(CXX) $(CXXFLAGS) -DSWIG -c $< -o $@ -I $(INC_PATH) -I $(SWIG_PATH)

$(PYGENN_PATH):
	mkdir -p $@

# Generate swig interfaces
prepare_swig_config: 
	python $(GENN_PATH)/lib/swig/generate_swig_interfaces.py $(GENN_PATH)/lib

# Build libgenn as shared library
$(LIBGENN_PYGENN): $(LIBGENN_PATH) $(PYGENN_CUSTOM_OBJ) $(LIBGENN_OBJ)
	$(CXX) -shared -o $@ $(LIBGENN_OBJ) $(PYGENN_CUSTOM_OBJ)

clean_pygenn: clean_libgenn
	rm -rf $(PYGENN_PATH)
	rm $(PYGENN_SWIG_FILE_PATH)
	rm $(SWIG_PATH)/SharedLibraryModel.i
	rm $(SWIG_PATH)/StlContainers.i
	rm $(SWIG_PATH)/currentSource*
	rm $(SWIG_PATH)/CurrentSourceModels.i
	rm $(SWIG_PATH)/newNeuron*
	rm $(SWIG_PATH)/NeuronModels.i
	rm $(SWIG_PATH)/newPostsynaptic*
	rm $(SWIG_PATH)/PostsynapticModels.i
	rm $(SWIG_PATH)/newWeightUpdate*
	rm $(SWIG_PATH)/WeightUpdateModels.i
	rm $(SWIG_PATH)/initVarSnippet*
	rm $(SWIG_PATH)/InitVarSnippet.i
