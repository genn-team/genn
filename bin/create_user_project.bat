ECHO OFF

REM Read project name and hence code directory from first command line argument
SET PROJECT_FILE=%1.vcxproj
SET CODE_DIRECTORY=%1_CODE

REM Create a new GUID for project
FOR /f %%i IN ('uuidgen -c') DO SET GUID=%%i

REM throw project name parameter away
SHIFT
SET SOURCE_FILES=%1
:loop
SHIFT
IF [%1]==[] GOTO afterloop
SET SOURCE_FILES=%SOURCE_FILES% %1
goto loop
:afterloop

REM Write out MSBuild project
@ECHO ^<?xml version="1.0" encoding="utf-8"?^> > %PROJECT_FILE%
@ECHO ^<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> %PROJECT_FILE%
@ECHO   ^<ItemGroup Label="ProjectConfigurations"^> >> %PROJECT_FILE%
@ECHO     ^<ProjectConfiguration Include="Debug|x64"^> >> %PROJECT_FILE%
@ECHO       ^<Configuration^>Debug^</Configuration^> >> %PROJECT_FILE%
@ECHO       ^<Platform^>x64^</Platform^> >> %PROJECT_FILE%
@ECHO     ^</ProjectConfiguration^> >> %PROJECT_FILE%
@ECHO     ^<ProjectConfiguration Include="Release|x64"^> >> %PROJECT_FILE%
@ECHO       ^<Configuration^>Release^</Configuration^> >> %PROJECT_FILE%
@ECHO       ^<Platform^>x64^</Platform^> >> %PROJECT_FILE%
@ECHO     ^</ProjectConfiguration^> >> %PROJECT_FILE%
@ECHO   ^</ItemGroup^> >> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Label="Globals"^> >> %PROJECT_FILE%
@ECHO     ^<ProjectGuid^>{%GUID%}^</ProjectGuid^> >> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^> >> %PROJECT_FILE%
@ECHO   ^<ItemGroup^> >> %PROJECT_FILE%
FOR %%U IN (%SOURCE_FILES%) DO (
    @ECHO     ^<ClCompile Include="%%U" /^> >> %PROJECT_FILE%
)
@ECHO   ^</ItemGroup^> >> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" /^> >> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration"^> >> %PROJECT_FILE%
@ECHO     ^<ConfigurationType^>Application^</ConfigurationType^> >> %PROJECT_FILE%
@ECHO     ^<UseDebugLibraries^>true^</UseDebugLibraries^> >> %PROJECT_FILE%
@ECHO     ^<PlatformToolset^>$(DefaultPlatformToolset)^</PlatformToolset^> >> %PROJECT_FILE%
@ECHO     ^<CharacterSet^>MultiByte^</CharacterSet^> >> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^> >> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration"^> >> %PROJECT_FILE%
@ECHO     ^<ConfigurationType^>Application^</ConfigurationType^> >> %PROJECT_FILE%
@ECHO     ^<UseDebugLibraries^>false^</UseDebugLibraries^> >> %PROJECT_FILE%
@ECHO     ^<PlatformToolset^>$(DefaultPlatformToolset)^</PlatformToolset^> >> %PROJECT_FILE%
@ECHO     ^<WholeProgramOptimization^>true^</WholeProgramOptimization^> >> %PROJECT_FILE%
@ECHO     ^<CharacterSet^>MultiByte^</CharacterSet^> >> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^> >> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" /^> >> %PROJECT_FILE%
@ECHO   ^<ImportGroup Label="ExtensionSettings"^> >> %PROJECT_FILE%
@ECHO   ^</ImportGroup^> >> %PROJECT_FILE%
@ECHO   ^<ImportGroup^> >> %PROJECT_FILE%
@ECHO     ^<Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" /^> >> %PROJECT_FILE%
@ECHO   ^</ImportGroup^> >> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Label="UserMacros" /^> >> %PROJECT_FILE%
@ECHO   ^<PropertyGroup^> >> %PROJECT_FILE%
@ECHO     ^<OutDir^>./^</OutDir^> >> %PROJECT_FILE%
@ECHO     ^<IntDir^>$(Platform)\$(Configuration)\^</IntDir^> >> %PROJECT_FILE%
@ECHO     ^<TargetName^>$(ProjectName)_$(Configuration)^</TargetName^> >> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^> >> %PROJECT_FILE%
@ECHO   ^<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'"^> >> %PROJECT_FILE%
@ECHO     ^<ClCompile^> >> %PROJECT_FILE%
@ECHO       ^<WarningLevel^>Level3^</WarningLevel^> >> %PROJECT_FILE%
@ECHO       ^<Optimization^>Disabled^</Optimization^> >> %PROJECT_FILE%
@ECHO       ^<SDLCheck^>true^</SDLCheck^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalIncludeDirectories^>%CODE_DIRECTORY%^</AdditionalIncludeDirectories^> >> %PROJECT_FILE%
@ECHO     ^</ClCompile^> >> %PROJECT_FILE%
@ECHO     ^<Link^> >> %PROJECT_FILE%
@ECHO       ^<GenerateDebugInformation^>true^</GenerateDebugInformation^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalDependencies^>runner_Debug.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalLibraryDirectories^>.^;%%(AdditionalLibraryDirectories)^</AdditionalLibraryDirectories^> >> %PROJECT_FILE%
@ECHO     ^</Link^> >> %PROJECT_FILE%
@ECHO   ^</ItemDefinitionGroup^> >> %PROJECT_FILE%
@ECHO   ^<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'"^> >> %PROJECT_FILE%
@ECHO     ^<ClCompile^> >> %PROJECT_FILE%
@ECHO       ^<WarningLevel^>Level3^</WarningLevel^> >> %PROJECT_FILE%
@ECHO       ^<Optimization^>MaxSpeed^</Optimization^> >> %PROJECT_FILE%
@ECHO       ^<FunctionLevelLinking^>true^</FunctionLevelLinking^> >> %PROJECT_FILE%
@ECHO       ^<IntrinsicFunctions^>true^</IntrinsicFunctions^> >> %PROJECT_FILE%
@ECHO       ^<SDLCheck^>true^</SDLCheck^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalIncludeDirectories^>%CODE_DIRECTORY%^</AdditionalIncludeDirectories^> >> %PROJECT_FILE%
@ECHO     ^</ClCompile^> >> %PROJECT_FILE%
@ECHO     ^<Link^> >> %PROJECT_FILE%
@ECHO       ^<GenerateDebugInformation^>true^</GenerateDebugInformation^> >> %PROJECT_FILE%
@ECHO       ^<EnableCOMDATFolding^>true^</EnableCOMDATFolding^> >> %PROJECT_FILE%
@ECHO       ^<OptimizeReferences^>true^</OptimizeReferences^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalDependencies^>runner_Release.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^> >> %PROJECT_FILE%
@ECHO       ^<AdditionalLibraryDirectories^>.^;%%(AdditionalLibraryDirectories)^</AdditionalLibraryDirectories^> >> %PROJECT_FILE%
@ECHO     ^</Link^> >> %PROJECT_FILE%
@ECHO   ^</ItemDefinitionGroup^> >> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" /^> >> %PROJECT_FILE%
@ECHO   ^<ImportGroup Label="ExtensionTargets"^> >> %PROJECT_FILE%
@ECHO   ^</ImportGroup^> >> %PROJECT_FILE%
@ECHO ^</Project^> >> %PROJECT_FILE%