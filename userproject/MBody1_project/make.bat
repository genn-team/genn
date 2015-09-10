REM Batch file to establish correct nmake build environment and build a target that is recognised in WINmakefile
REM Argument 1: name of the target
REM Argument 2,3: optional "DEBUG=1" to switch debugging mode on or "CPU_ONLY=1" to build a CPU (i.e. CUDA independent) version
ECHO OFF
WHERE /Q nmake.exe && GOTO themake

IF %PROCESSOR_ARCHITECTURE% EQU AMD64 (GOTO x64) ELSE (GOTO check2)

:check2
IF %PROCESSOR_ARCHITECTUREW64W32% EQU AMD64 (GOTO x64) ELSE (GOTO x86)

:x64
set ARCH=x64
GOTO finish;

:x86
set ARCH=x86
GOTO finish:

:finish
"%VS_PATH%\VC\vcvarsall.bat" %ARCH% 

:themake
nmake /nologo /f WINmakefile %1% %2% %3%
