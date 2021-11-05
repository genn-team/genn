ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM SET "BUILD_FLAGS=-c"
REM SET "MAKE_FLAGS=CPU_ONLY=1"

REM Push feature directories
PUSHD features

REM Loop through feature tests
FOR /D %%F IN (*) DO (
	ECHO Running test %%F
	
	REM Push feature directory
	PUSHD %%F
	
	REM Build model
	CALL genn-buildmodel.bat -l %BUILD_FLAGS% model.cc
	
	REM Build model
	msbuild "%%F.sln" /m /t:%%F /p:BuildProjectReferences=true /verbosity:minimal /p:Configuration=Release
	
	REM Run tests
	test.exe --test_output="xml:test_results.xml"

	REM pop directory
	POPD
)

REM pop features
POPD

REM double-check GeNN is built
msbuild ../genn.sln /m /t:single_threaded_cpu_backend /verbosity:minimal /p:Configuration=Release

REM build unit tests
msbuild tests.sln /m /verbosity:minimal /p:Configuration=Release

PUSHD unit

REM run tests
unit_Release.exe --gtest_output="xml:test_results_unit.xml"

REM pop unit
POPD