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
nmake /f WINmakefile %1%