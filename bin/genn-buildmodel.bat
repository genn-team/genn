@echo off
setlocal ENABLEDELAYEDEXPANSION
SET GENN_PATH=%~DP0
goto :genn_begin

:genn_help
rem :: display genn-buildmodel.bat help
echo genn-buildmodel.bat script usage:
echo genn-buildmodel.bat [cdho] model
echo -c             only generate simulation code for the CPU
echo -d             enables the debugging mode
echo -h             shows this help message
echo -o outpath     changes the output directory
echo -i includepath add additional include directories (seperated by semicolons)
goto :eof

:genn_begin
rem :: define genn-buildmodel.bat options separated by spaces
rem :: -<option>:              option
rem :: -<option>:""            option with argument
rem :: -<option>:"<default>"   option with argument and default value
set "OPTIONS=-o:"%CD%" -i:"" -d: -c: -h:"
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
if not defined MODEL (
    echo genn-buildmodel.bat: error 2: no model file given
    goto :eof
)
for /f %%I in ("%-o%") do set "-o=%%~fI"
for /f %%I in ("%-i%") do set "-i=%%~fI"
for /f %%I in ("%MODEL%") do set "MACROS=/p:ModelFile=%%~fI /p:GeneratePath=%-o% /p:BuildModelInclude=%-i%"

if defined -d (
	set "BACKEND_MACROS= /p:Configuration=Debug"
	if defined -c (
		set "BACKEND_PROJECT=single_threaded_cpu_backend"
		set "MACROS=%MACROS% /p:Configuration=Debug"
		set GENERATOR=.\generator_Debug.exe
	) else (
		set "BACKEND_PROJECT=cuda_backend"
		set "MACROS=%MACROS% /p:Configuration=Debug_CUDA"
		set GENERATOR=.\generator_Debug_CUDA.exe
	)    
) else (
	set "BACKEND_MACROS= /p:Configuration=Release"
	if defined -c (
		set "BACKEND_PROJECT=single_threaded_cpu_backend"
		set "MACROS=%MACROS% /p:Configuration=Release"
		set GENERATOR=.\generator_Release.exe
	) else (
		set "BACKEND_PROJECT=cuda_backend"
		set "MACROS=%MACROS% /p:Configuration=Release_CUDA"
		set GENERATOR=.\generator_Release_CUDA.exe
	)
)


rem :: build backend
msbuild "%GENN_PATH%..\genn.sln" /m /verbosity:minimal /t:%BACKEND_PROJECT% %BACKEND_MACROS% /p:BuildProjectReferences=true && (
	echo Successfully built GeNN
) || (
	echo Unable to build GeNN
	goto :eof
)


rem :: build generator
msbuild "%GENN_PATH%..\src\genn\generator\generator.vcxproj" /m /verbosity:minimal %MACROS%&& (
	echo Successfully built code generator
) || (
	echo Unable to build code generator
	goto :eof
)

if defined -d (
	devenv /debugexe "%GENERATOR%" "%-o%"
) else (
	"%GENERATOR%" "%-o%"
)

echo Model build complete
