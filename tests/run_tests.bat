ECHO OFF

REM double-check GeNN is built
msbuild ../genn.sln /m /t:single_threaded_cpu_backend /verbosity:minimal /p:Configuration=Release

REM build unit tests
msbuild tests.sln /m /verbosity:minimal /p:Configuration=Release

PUSHD unit

REM run tests
unit_Release.exe --gtest_output="xml:test_results_unit.xml"

REM pop unit
POPD
