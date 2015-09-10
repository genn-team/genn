@echo off

set MODELPATH=%cd%
echo model path: %MODELPATH%
set MODELNAME=%1
echo model name: %MODELNAME%
set /a k=0
set DBGMODE=0
set EXTRA_DEF=
for %%op in (%*) do (
    if %k > 0 (
REM op=$(echo $op | tr [a-z] [A-Z])
	if "%op%" == "DEBUG=1" (
	    set DBGMODE=1
	)
	if "%op%" == "CPU_ONLY=1" (
	    set EXTRA_DEF=CPU_ONLY
	)   
    )
    set /a k=!k!+1
)

if "%EXTRA_DEF%" != "" (
    set EXTRA_DEF=-D$EXTRA_DEF
)

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

cd /d "%GENN_PATH%\lib"
nmake /nologo /f WINmakefile clean
if "%DBGMODE%"=="0" (
  nmake /nologo /f WINmakefile MODEL="%MODELPATH%\%MODELNAME%.cc"
  bin\generateALL.exe %MODELPATH%
) else (
  echo "Debugging mode ON"
  nmake /nologo /f WINmakefile DEBUG=1 MODEL="%MODELPATH%\%MODELNAME%.cc"
  devenv /debugexe bin\generateALL.exe %MODELPATH%
)
cd /d "%MODELPATH%"

echo Model build complete ...
