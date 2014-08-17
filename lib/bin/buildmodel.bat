@echo off

set MODELPATH=%cd%
echo "model path:" %MODELPATH%
set MODELNAME=%1
echo "model name:" %MODELNAME%
set DBGMODE=%2
if "%DBGMODE%"=="" set DBGMODE=0

copy "%MODELPATH%\%MODELNAME%.cc" "%GeNNPATH%\lib\src\currentModel.cc"
cd "%GeNNPATH%\lib"

if "%DBGMODE%"=="0" (
  nmake /nologo /f WINmakefile clean
  nmake /nologo /f WINmakefile
  bin\generateALL %MODELPATH%
) else (
  echo "Debugging mode ON"
  rem "Windows debug commands will go here ..."
)

cd "%MODELPATH%"
echo Model build complete ...
