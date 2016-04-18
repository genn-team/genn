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

# Global CUDA compiler settings
!IFNDEF CPU_ONLY
NVCC			="$(CUDA_PATH)\bin\nvcc.exe"
!ENDIF
!IFNDEF DEBUG
NVCCFLAGS		=$(NVCCFLAGS) --compiler-options $(OPTIMIZATIONFLAGS)
!ELSE
NVCCFLAGS		=$(NVCCFLAGS) -g -G
!ENDIF

# Global C++ compiler settings
!IFNDEF CPU_ONLY
CXXFLAGS		=$(CXXFLAGS) /nologo /EHsc
!ELSE
CXXFLAGS		=$(CXXFLAGS) /nologo /EHsc /DCPU_ONLY
!ENDIF
!IFNDEF DEBUG
CXXFLAGS		=$(CXXFLAGS) $(OPTIMIZATIONFLAGS)
!ELSE
CXXFLAGS		=$(CXXFLAGS) /debug /Zi /Od
!ENDIF

# Global include and link flags
!IFNDEF CPU_ONLY
INCLUDE_FLAGS		=/I"$(GENN_PATH)\lib\include" /I"$(GENN_PATH)\userproject\include" /I"$(CUDA_PATH)\include"
!IF "$(PROCESSOR_ARCHITECTURE)" == "AMD64"
LINK_FLAGS		="$(CUDA_PATH)\lib\x64\cudart.lib" "$(CUDA_PATH)\lib\x64\cuda.lib"
!ELSEIF "$(PROCESSOR_ARCHITEW6432)" == "AMD64"
LINK_FLAGS		="$(CUDA_PATH)\lib\x64\cudart.lib" "$(CUDA_PATH)\lib\x64\cuda.lib"
!ELSE
LINK_FLAGS		="$(CUDA_PATH)\lib\Win32\cudart.lib" "$(CUDA_PATH)\lib\Win32\cuda.lib"
!ENDIF
!ELSE
INCLUDE_FLAGS		=/I"$(GENN_PATH)\lib\include" /I"$(GENN_PATH)\userproject\include"
!ENDIF

# An auto-generated file containing your cuda device's compute capability
!IF EXIST(sm_version.mk)
!INCLUDE sm_version.mk
!ENDIF

# Infer object file names from source file names
OBJECTS		=$(SOURCES:.cc=.obj)
OBJECTS		=$(OBJECTS:.cpp=.obj)
OBJECTS		=$(OBJECTS:.cu=.obj)

# Target rules
.SUFFIXES: .cu

all: $(EXECUTABLE)

.cc.obj:
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $** /Fo$@ /c

.cpp.obj:
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $** /Fo$@ /c

!IFNDEF CPU_ONLY
.cu.obj:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS:/I=-I) $(GENCODE_FLAGS) $** /Fo$@ -c
!ENDIF

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) /Fe$@ $(LINK_FLAGS)

clean:
	-del $(EXECUTABLE) *.obj *.ilk *.pdb 2>nul

purge: clean
	-del sm_version.mk 2>nul
	-rd /s /q *_CODE 2>nul
