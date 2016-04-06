@ECHO off
setlocal EnableDelayedExpansion

rem : parse command options
set options=


set INITIAL_PATH="%CD%"
set OUTPUT_PATH="%INITIAL_PATH%"





set MODEL="%1"


set k=0
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







if "%MODEL%"=="" (
    echo genn-buildmodel.bat: error 2: no model file given
    exit 2
)

rem : checking GENN_PATH is defined
if "%GENN_PATH%"=="" (
    echo genn-buildmodel.bat: error 3: GENN_PATH is not defined
    exit 3
)

rem : generate model code
cd /d "%OUTPUT_PATH%"
nmake clean /nologo /f "%GENN_PATH%\lib\src\WINmakefile"
if "%DEBUG%"=="1" (
    echo debugging mode ON
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" MODEL="%MODEL%" CPU_ONLY=%CPU_ONLY% DEBUG=%DEBUG%
    devenv /debugexe .\generateALL.exe "%OUTPUT_PATH%"
) else (
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" MODEL="%MODEL%" CPU_ONLY=%CPU_ONLY%
    .\generateALL.exe "%OUTPUT_PATH%"
)

echo model build complete
goto :eof

rem : display genn-buildmodel.bat help and quit
:genn_help
    echo === genn-buildmodel.bat script usage ===
    echo genn-buildmodel.bat [cdho] <model>
    echo -c only generate simulation code for the CPU
    echo -d enables the debugging mode
    echo -h shows this help message
    echo -o <path> changes the output directory to <path>
    goto :eof
