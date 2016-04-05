@echo off

set FLAGS="$PWD/$1.cc"




set k=0
setlocal ENABLEDELAYEDEXPANSION
for %%x in (%*) do (
    set /a k+=1
    set "argv[!k!]=%%~x"
)
for /l %%i in (1,1,%k%) do (
    if "!argv[%%i]!"=="DEBUG" (
      set /a j=%%i+1
      set /a DBGMODE="argv[!j!]"
    )  
    if "!argv[%%i]!"=="CPU_ONLY" (
      set /a j=%%i+1
      set /a CPU_ONLY="argv[!j!]"
    )
) 
for /f %%a in ('set DBGMODE^&set CPU_ONLY') do (if "!"=="" endlocal)&set "%%a"




echo warning: buildmodel.bat has been depreciated!
echo please use the new genn-buildmodel.bat script in future
echo the equivalent genn-buildmodel.bat command is:
echo genn-buildmodel.bat %FLAGS%

genn-buildmodel.bat %FLAGS%
