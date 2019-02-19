ECHO OFF

REM Read project name and hence code directory from first command line argument
SET PROJECT_NAME=%1.vcxproj
SET CODE_DIRECTORY=%1_CODE

REM Write out MSBuild project
@ECHO ^<?xml version="1.0" encoding="utf-8"?^> > %PROJECT_NAME%
@ECHO ^<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> %PROJECT_NAME%
@ECHO   ^<ItemGroup Label="ProjectConfigurations"^> >> %PROJECT_NAME%
@ECHO     ^<ProjectConfiguration Include="Debug|x64"^> >> %PROJECT_NAME%
@ECHO       ^<Configuration^>Debug^</Configuration^> >> %PROJECT_NAME%
@ECHO       ^<Platform^>x64^</Platform^> >> %PROJECT_NAME%
@ECHO     ^</ProjectConfiguration^> >> %PROJECT_NAME%
@ECHO     ^<ProjectConfiguration Include="Release|x64"^> >> %PROJECT_NAME%
@ECHO       ^<Configuration^>Release^</Configuration^> >> %PROJECT_NAME%
@ECHO       ^<Platform^>x64^</Platform^> >> %PROJECT_NAME%
@ECHO     ^</ProjectConfiguration^> >> %PROJECT_NAME%
@ECHO   ^</ItemGroup^> >> %PROJECT_NAME%
@ECHO   ^<PropertyGroup Label="Globals"^> >> %PROJECT_NAME%
@ECHO     ^<ProjectGuid^>{85FBA02A-804A-49CB-8374-892CCBC49A54}^</ProjectGuid^> >> %PROJECT_NAME%
@ECHO   ^</PropertyGroup^> >> %PROJECT_NAME%
@ECHO   ^<ItemGroup^> >> %PROJECT_NAME%
FOR %%U IN (%*) DO (
    @ECHO     ^<ClCompile Include="%%U" /^> >> %PROJECT_NAME%
)
@ECHO   ^</ItemGroup^> >> %PROJECT_NAME%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" /^> >> %PROJECT_NAME%
@ECHO   ^<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration"^> >> %PROJECT_NAME%
@ECHO     ^<ConfigurationType^>Application^</ConfigurationType^> >> %PROJECT_NAME%
@ECHO     ^<UseDebugLibraries^>true^</UseDebugLibraries^> >> %PROJECT_NAME%
@ECHO     ^<PlatformToolset^>$(DefaultPlatformToolset)^</PlatformToolset^> >> %PROJECT_NAME%
@ECHO     ^<CharacterSet^>MultiByte^</CharacterSet^> >> %PROJECT_NAME%
@ECHO   ^</PropertyGroup^> >> %PROJECT_NAME%
@ECHO   ^<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration"^> >> %PROJECT_NAME%
@ECHO     ^<ConfigurationType^>Application^</ConfigurationType^> >> %PROJECT_NAME%
@ECHO     ^<UseDebugLibraries^>false^</UseDebugLibraries^> >> %PROJECT_NAME%
@ECHO     ^<PlatformToolset^>$(DefaultPlatformToolset)^</PlatformToolset^> >> %PROJECT_NAME%
@ECHO     ^<WholeProgramOptimization^>true^</WholeProgramOptimization^> >> %PROJECT_NAME%
@ECHO     ^<CharacterSet^>MultiByte^</CharacterSet^> >> %PROJECT_NAME%
@ECHO   ^</PropertyGroup^> >> %PROJECT_NAME%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" /^> >> %PROJECT_NAME%
@ECHO   ^<ImportGroup Label="ExtensionSettings"^> >> %PROJECT_NAME%
@ECHO   ^</ImportGroup^> >> %PROJECT_NAME%
@ECHO   ^<ImportGroup^> >> %PROJECT_NAME%
@ECHO     ^<Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" /^> >> %PROJECT_NAME%
@ECHO   ^</ImportGroup^> >> %PROJECT_NAME%
@ECHO   ^<PropertyGroup Label="UserMacros" /^> >> %PROJECT_NAME%
@ECHO   ^<PropertyGroup^> >> %PROJECT_NAME%
@ECHO     ^<OutDir^>./^</OutDir^> >> %PROJECT_NAME%
@ECHO     ^<IntDir^>$(Platform)\$(Configuration)\^</IntDir^> >> %PROJECT_NAME%
@ECHO     ^<TargetName^>$(ProjectName)_$(Configuration)^</TargetName^> >> %PROJECT_NAME%
@ECHO   ^</PropertyGroup^> >> %PROJECT_NAME%
@ECHO   ^<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'"^> >> %PROJECT_NAME%
@ECHO     ^<ClCompile^> >> %PROJECT_NAME%
@ECHO       ^<WarningLevel^>Level3^</WarningLevel^> >> %PROJECT_NAME%
@ECHO       ^<Optimization^>Disabled^</Optimization^> >> %PROJECT_NAME%
@ECHO       ^<SDLCheck^>true^</SDLCheck^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalIncludeDirectories^>%CODE_DIRECTORY%^</AdditionalIncludeDirectories^> >> %PROJECT_NAME%
@ECHO     ^</ClCompile^> >> %PROJECT_NAME%
@ECHO     ^<Link^> >> %PROJECT_NAME%
@ECHO       ^<GenerateDebugInformation^>true^</GenerateDebugInformation^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalDependencies^>runner_Debug.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalLibraryDirectories^>.^;%%(AdditionalLibraryDirectories)^</AdditionalLibraryDirectories^> >> %PROJECT_NAME%
@ECHO     ^</Link^> >> %PROJECT_NAME%
@ECHO   ^</ItemDefinitionGroup^> >> %PROJECT_NAME%
@ECHO   ^<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'"^> >> %PROJECT_NAME%
@ECHO     ^<ClCompile^> >> %PROJECT_NAME%
@ECHO       ^<WarningLevel^>Level3^</WarningLevel^> >> %PROJECT_NAME%
@ECHO       ^<Optimization^>MaxSpeed^</Optimization^> >> %PROJECT_NAME%
@ECHO       ^<FunctionLevelLinking^>true^</FunctionLevelLinking^> >> %PROJECT_NAME%
@ECHO       ^<IntrinsicFunctions^>true^</IntrinsicFunctions^> >> %PROJECT_NAME%
@ECHO       ^<SDLCheck^>true^</SDLCheck^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalIncludeDirectories^>%CODE_DIRECTORY%^</AdditionalIncludeDirectories^> >> %PROJECT_NAME%
@ECHO     ^</ClCompile^> >> %PROJECT_NAME%
@ECHO     ^<Link^> >> %PROJECT_NAME%
@ECHO       ^<GenerateDebugInformation^>true^</GenerateDebugInformation^> >> %PROJECT_NAME%
@ECHO       ^<EnableCOMDATFolding^>true^</EnableCOMDATFolding^> >> %PROJECT_NAME%
@ECHO       ^<OptimizeReferences^>true^</OptimizeReferences^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalDependencies^>runner_Release.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^> >> %PROJECT_NAME%
@ECHO       ^<AdditionalLibraryDirectories^>.^;%%(AdditionalLibraryDirectories)^</AdditionalLibraryDirectories^> >> %PROJECT_NAME%
@ECHO     ^</Link^> >> %PROJECT_NAME%
@ECHO   ^</ItemDefinitionGroup^> >> %PROJECT_NAME%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" /^> >> %PROJECT_NAME%
@ECHO   ^<ImportGroup Label="ExtensionTargets"^> >> %PROJECT_NAME%
@ECHO   ^</ImportGroup^> >> %PROJECT_NAME%
@ECHO ^</Project^> >> %PROJECT_NAME%