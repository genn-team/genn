##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##            	Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------


# Makefile include for all GeNN projects
# This is a Windows Makefile, to be used by the MS nmake build system
#--------------------------------------------------------------------

# Global C++ and CUDA compiler settings.
NVCC		="$(CUDA_PATH)\bin\nvcc.exe"
NVCCFLAGS	=$(NVCCFLAGS)
CXXFLAGS	=$(CXXFLAGS) /nologo /EHsc

# Global include flags and link flags.
INCLUDE_FLAGS	= /I"$(CUDA_PATH)\include" /I$(CUDA_PATH)\samples\common\inc /I"$(GENN_PATH)\lib\include" /I"$(GENN_PATH)\userproject\include" $(EXTRA_INCLUDE) 
!IF "$(PROCESSOR_ARCHITECTURE)" == "AMD64"
LINK_FLAGS	="$(CUDA_PATH)\lib\x64\cudart.lib"
!ELSEIF "$(PROCESSOR_ARCHITEW6432)" == "AMD64"
LINK_FLAGS	="$(CUDA_PATH)\lib\x64\cudart.lib"
!ELSE
LINK_FLAGS	="$(CUDA_PATH)\lib\Win32\cudart.lib"
!ENDIF

# An auto-generated file containing your cuda device's compute capability.
!INCLUDE sm_version.mk

# Infer object file names from source file names.
OBJECTS		=$(SOURCES:.cc=.obj)
OBJECTS		=$(OBJECTS:.cpp=.obj)
OBJECTS		=$(OBJECTS:.cu=.obj)

# Target rules.
.SUFFIXES: .cu

all: $(EXECUTABLE)

.cc.obj:
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< /Fo$@ /c

.cpp.obj:
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< /Fo$@ /c

.cu.obj:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS:/I=-I) $(GENCODE_FLAGS) $< /Fo$@ -c

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LINK_FLAGS) $(OBJECTS) /Fe$@

clean:
	-del $(EXECUTABLE) *.obj *.ilk *.pdb 2>nul

purge: clean
	-del sm_version.mk 2>nul
	-rd /s /q *_CODE 2>nul
