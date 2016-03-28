@echo off

set PROJECT_PATH="%cd%"

set MODELNAME="%1"



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
echo debug mode: %DBGMODE%
echo cpu only: %CPU_ONLY%





if "%GENN_PATH%"=="" (
    echo error: GENN_PATH is not defined
    exit
)




cd /d "%PROJECT_PATH%"
nmake clean /nologo /f "%GENN_PATH%\lib\src\WINmakefile"
if "%DEBUG_MODE%"=="1" (
    echo "debugging mode ON"
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" MODEL="%MODEL%" CPU_ONLY=%CPU_ONLY% DEBUG_MODE=%DEBUG_MODE%
    devenv /debugexe .\generateALL.exe "%PROJECT_PATH%"
) else (
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" MODEL="%MODEL%" CPU_ONLY=%CPU_ONLY%
    .\generateALL.exe "%PROJECT_PATH%"
)

echo Model build complete
