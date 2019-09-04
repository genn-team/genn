@ECHO OFF

FOR /d %%# IN (*) DO (
    IF NOT EXIST "%%~f#\*.sln" ECHO %%~f#
)