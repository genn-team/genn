@echo off
setlocal ENABLEDELAYEDEXPANSION
goto :genn_begin

:genn_help
rem :: display genn-buildmodel.bat help
echo === genn-buildmodel.bat script usage ===
echo genn-buildmodel.bat [cdho] model
echo -c            only generate simulation code for the CPU
echo -d            enables the debugging mode
echo -h            shows this help message
echo -o outpath    changes the output directory
goto :eof

:genn_begin
rem :: define genn-buildmodel.bat options separated by spaces
rem :: -<option>:              option
rem :: -<option>:""            option with argument
rem :: -<option>:"<default>"   option with argument and default value
set "OPTIONS=-o:"%CD%" -d: -c: -h:"
for %%O in (%OPTIONS%) do for /f "tokens=1,* delims=:" %%A in ("%%O") do set "%%A=%%~B"

:genn_option
rem :: parse command options
set "OPT=%~1"
if defined OPT (
    set "test=!OPTIONS:*%~1:=! "
    if "!test!"=="!OPTIONS! " (
        if "!OPT:~0,1!"=="-" (
            echo genn-buildmodel.bat: error: invalid option: !OPT!
            goto :genn_help
        )
        set "MODEL=!OPT!"
        shift /1
        goto :genn_option
    )
    if "!test:~0,1!"==" " (
        set "!OPT!=1"
    ) else (
        set "!OPT!=%~2"
        shift /1
    )
    shift /1
    goto :genn_option
)

rem :: command options logic
if defined -h goto :genn_help
if not defined GENN_PATH (
    echo genn-buildmodel.bat: error 1: GENN_PATH is not defined
    goto :eof
)
if not defined MODEL (
    echo genn-buildmodel.bat: error 2: no model file given
    goto :eof
)
for /f %%I in ("%-o%") do set "-o=%%~fI"
for /f %%I in ("%MODEL%") do set "MACROS=MODEL=%%~fI"
if defined -c set "MACROS=%MACROS% CPU_ONLY=1"
if defined -d set "MACROS=%MACROS% DEBUG=1"

rem :: generate model code
cd /d "%-o%"
nmake clean /nologo /f "%GENN_PATH%\lib\src\WINmakefile"
if defined -d (
    echo debugging mode ON
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" %MACROS%
    devenv /debugexe .\generateALL.exe "%-o%"
) else (
    nmake /nologo /f "%GENN_PATH%\lib\src\WINmakefile" %MACROS%
    .\generateALL.exe "%-o%"
)

echo model build complete
goto :eof
