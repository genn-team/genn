@echo off

set MODELPATH=%cd%
echo model path: %MODELPATH%
set MODELNAME=%1
echo model name: %MODELNAME%
set DBGMODE=%2
if "%DBGMODE%"=="" set DBGMODE=0

if "%GENN_PATH%"=="" (
  if "%GeNNPATH%"=="" (
    echo ERROR: Environment variable 'GENN_PATH' has not been defined. Quitting...
    exit
  )
  echo Environment variable 'GeNNPATH' will be replaced by 'GENN_PATH' in future GeNN releases.
  set GENN_PATH=%GeNNPATH%
)

copy "%MODELPATH%\%MODELNAME%.cc" "%GENN_PATH%\lib\src\currentModel.cc"
cd "%GENN_PATH%\lib"

nmake /nologo /f WINmakefile clean
if "%DBGMODE%"=="0" (
  nmake /nologo /f WINmakefile
  bin\generateALL.exe %MODELPATH%
) else (
  echo "Debugging mode ON"
  nmake /nologo /f WINmakefile DEBUG=1
  devenv /debugexe bin\generateALL.exe %MODELPATH%
)

cd "%MODELPATH%"
echo Model build complete ...
