@echo off

set MODELPATH=%cd%
echo model path: %MODELPATH%
set MODELNAME=%1
echo model name: %MODELNAME%
set DBGMODE=%2
if "%DBGMODE%"=="" set DBGMODE=0

copy "%MODELPATH%\%MODELNAME%.cc" "%GENNPATH%\lib\src\currentModel.cc"
cd "%GENNPATH%\lib"

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
