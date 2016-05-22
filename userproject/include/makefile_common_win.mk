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


# Makefile include for all GeNN example projects
# This is a Windows Makefile, to be used by the MS nmake build system
#--------------------------------------------------------------------

# Global CUDA compiler settings
!IFNDEF CPU_ONLY
NVCC                    ="$(CUDA_PATH)\bin\nvcc.exe"
!ENDIF
!IFDEF DEBUG
NVCCFLAGS               =$(NVCCFLAGS) -g -G
!ELSE
NVCCFLAGS               =$(NVCCFLAGS) $(NVCC_OPTIMIZATIONFLAGS) -Xcompiler "$(OPTIMIZATIONFLAGS)"
!ENDIF

# Global C++ compiler settings
!IFNDEF CPU_ONLY
CXXFLAGS                =$(CXXFLAGS) /nologo /EHsc
!ELSE
CXXFLAGS                =$(CXXFLAGS) /nologo /EHsc /DCPU_ONLY
!ENDIF
!IFDEF DEBUG
CXXFLAGS                =$(CXXFLAGS) /debug /Zi /Od
!ELSE
CXXFLAGS                =$(CXXFLAGS) $(OPTIMIZATIONFLAGS)
!ENDIF

# Global include and link flags
!IFNDEF CPU_ONLY
INCLUDE_FLAGS           =/I"$(GENN_PATH)\lib\include" /I"$(GENN_PATH)\userproject\include" /I"$(CUDA_PATH)\include"
!IF "$(PROCESSOR_ARCHITECTURE)" == "AMD64" || "$(PROCESSOR_ARCHITEW6432)" == "AMD64"
LINK_FLAGS              ="$(GENN_PATH)\lib\lib\genn.lib" "$(CUDA_PATH)\lib\x64\cudart.lib" "$(CUDA_PATH)\lib\x64\cuda.lib"
!ELSE
LINK_FLAGS              ="$(GENN_PATH)\lib\lib\genn.lib" "$(CUDA_PATH)\lib\Win32\cudart.lib" "$(CUDA_PATH)\lib\Win32\cuda.lib"
!ENDIF
!ELSE
INCLUDE_FLAGS           =/I"$(GENN_PATH)\lib\include" /I"$(GENN_PATH)\userproject\include"
LINK_FLAGS              ="$(GENN_PATH)\lib\lib\genn.lib"
!ENDIF

# An auto-generated file containing your cuda device's compute capability
!IF EXIST(sm_version.mk)
!INCLUDE sm_version.mk
!ENDIF

# Infer object file names from source file names
!IFNDEF SIM_CODE
!ERROR You must define SIM_CODE=<model>_CODE in the Makefile or NMAKE command.
!ENDIF
OBJECTS                 =$(SOURCES:.cc=.obj) $(SIM_CODE)\runner.obj
OBJECTS                 =$(OBJECTS:.cpp=.obj)
OBJECTS                 =$(OBJECTS:.cu=.obj)


# Target rules
.SUFFIXES: .cu

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) /Fe$@ $(OBJECTS) $(LINK_FLAGS)

$(SIM_CODE)\runner.obj:
	cd $(SIM_CODE) && nmake /nologo

.cc.obj:
	$(CXX) $(CXXFLAGS) /c /Fo$@ $** $(INCLUDE_FLAGS)

.cpp.obj:
	$(CXX) $(CXXFLAGS) /c /Fo$@ $** $(INCLUDE_FLAGS)

!IFNDEF CPU_ONLY
.cu.obj:
	$(NVCC) $(NVCCFLAGS) /c /Fo$@ $** $(INCLUDE_FLAGS:/I=-I)
!ENDIF

clean:
	-del $(EXECUTABLE) *.obj *.ilk *.pdb 2>nul
	cd $(SIM_CODE) && nmake clean /nologo

purge: clean
	-del sm_version.mk 2>nul
	-rd /s /q $(SIM_CODE) 2>nul
