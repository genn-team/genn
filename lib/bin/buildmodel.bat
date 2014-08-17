@echo off

set MODELPATH=%cd%
echo "model path:" %MODELPATH%
set MODELNAME=%1
echo "model name:" %MODELNAME%
set DBGMODE=%2 # 1 if debugging, 0 if release

copy "%MODELPATH%\%MODELNAME%.cc" "%GeNNPATH%\lib\src\currentModel.cc"
cd "$GeNNPATH\lib"

if %DBGMODE% equ 1 (
  echo "Debugging mode ON"

  rem ---debug launch command goes here---

) else (
  nmake /nologo /f WINmakefile clean
  nmake /nologo /f WINmakefile
  bin\generateALL %MODELPATH%
)
cd "%MODELPATH%"

echo "Model build complete ..."
