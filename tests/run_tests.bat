ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

SET "BUILD_FLAGS=-c"
SET "MAKE_FLAGS=CPU_ONLY=1"

REM suffix to apply to final folder of feature directory to get path
SET "SUFFIX=_CODE"

REM Loop through feature tests
FOR /D %%F IN ("features\*") DO (
	ECHO Running test %%F
	
	REM Push feature directory
	PUSHD %%F
	
	REM Clean
	nmake -f WINMakefile clean SIM_CODE=%%~nxF%SUFFIX%
	
	REM Build model
	genn-buildmodel.bat %BUILD_FLAGS% model.cc
	
	REM **YUCK** genn-build model seems to mess up path
	CD %%F
	
	REM Build model
	nmake -f WINMakefile SIM_CODE=%%~nxF%SUFFIX% %MAKE_FLAGS%
	
	REM Run tests
    test.exe --gtest_output="xml:test_results.xml"
	
	REM pop directory
	POPD
)