ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

:loop
REM If there are no more arguments, exit loop
IF [%1]==[] (
    GOTO afterloop
)

REM If this argument is specifying an include directory
IF [%1]==[-i] (
    REM Perform additional shift to one at end of loop to skip over both 
    SET "INCLUDE_DIRS=!INCLUDE_DIRS!;%2"
    SHIFT
) ELSE (
    REM Otherwise, if this argument is enabling user project include directory
    IF [%1]==[-u] (
        SET "INCLUDE_USERPROJECT=1"
    ) ELSE (
        REM Otherwise, if no project name is yet set
        IF "%PROJECT_NAME%"=="" (
            SET "PROJECT_NAME=%1"
        ) ELSE (
            SET "SOURCE_FILES=!SOURCE_FILES! %1"
        )
    )
)

REM Shift out processed argument
SHIFT
GOTO loop
:afterloop

SET PROJECT_FILE=%PROJECT_NAME%.vcxproj
SET SOLUTION_FILE=%PROJECT_NAME%.sln
SET CODE_DIRECTORY=%PROJECT_NAME%_CODE
SET RUNNER_GUID_FILE=runner_guid.txt

REM Create a new GUID for user project
FOR /f %%i IN ('uuidgen -c') DO SET USER_GUID=%%i

REM If runner GUID exists, read it from file
IF EXIST %RUNNER_GUID_FILE% (
    FOR /f %%i IN (%RUNNER_GUID_FILE%) DO SET RUNNER_GUID=%%i
REM Otherwise generate a new GUID and write it to file
) ELSE (
    FOR /f %%i IN ('uuidgen -c') DO SET RUNNER_GUID=%%i
    @ECHO !RUNNER_GUID! > %RUNNER_GUID_FILE%
)

REM Write out MSBuild project
@ECHO ^<?xml version="1.0" encoding="utf-8"?^>> %PROJECT_FILE%
@ECHO ^<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^>>> %PROJECT_FILE%
@ECHO   ^<ItemGroup Label="ProjectConfigurations"^>>> %PROJECT_FILE%
@ECHO     ^<ProjectConfiguration Include="Debug|x64"^>>> %PROJECT_FILE%
@ECHO       ^<Configuration^>Debug^</Configuration^>>> %PROJECT_FILE%
@ECHO       ^<Platform^>x64^</Platform^>>> %PROJECT_FILE%
@ECHO     ^</ProjectConfiguration^>>> %PROJECT_FILE%
@ECHO     ^<ProjectConfiguration Include="Release|x64"^>>> %PROJECT_FILE%
@ECHO       ^<Configuration^>Release^</Configuration^>>> %PROJECT_FILE%
@ECHO       ^<Platform^>x64^</Platform^>>> %PROJECT_FILE%
@ECHO     ^</ProjectConfiguration^>>> %PROJECT_FILE%
@ECHO   ^</ItemGroup^>>> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Label="Globals"^>>> %PROJECT_FILE%
@ECHO     ^<ProjectGuid^>{%USER_GUID%}^</ProjectGuid^>>> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^>>> %PROJECT_FILE%
@ECHO   ^<ItemGroup^>>> %PROJECT_FILE%
FOR %%S IN (%SOURCE_FILES%) DO (
    @ECHO     ^<ClCompile Include="%%S" /^>>> %PROJECT_FILE%
)
@ECHO   ^</ItemGroup^>>> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" /^>>> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Label="Configuration"^>>> %PROJECT_FILE%
@ECHO     ^<ConfigurationType^>Application^</ConfigurationType^>>> %PROJECT_FILE%
@ECHO     ^<UseDebugLibraries  Condition="'$(Configuration)'=='Debug'"^>true^</UseDebugLibraries^>>> %PROJECT_FILE%
@ECHO     ^<PlatformToolset^>$(DefaultPlatformToolset)^</PlatformToolset^>>> %PROJECT_FILE%
@ECHO     ^<WholeProgramOptimization Condition="'$(Configuration)'=='Release'"^>true^</WholeProgramOptimization^>>> %PROJECT_FILE%
@ECHO     ^<CharacterSet^>MultiByte^</CharacterSet^>>> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^>>> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" /^>>> %PROJECT_FILE%
@ECHO   ^<ImportGroup Label="ExtensionSettings"^>>> %PROJECT_FILE%
@ECHO   ^</ImportGroup^>>> %PROJECT_FILE%
@ECHO   ^<ImportGroup^>>> %PROJECT_FILE%
@ECHO     ^<Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" /^>>> %PROJECT_FILE%
@ECHO   ^</ImportGroup^>>> %PROJECT_FILE%
@ECHO   ^<PropertyGroup Label="UserMacros" /^>>> %PROJECT_FILE%
@ECHO   ^<PropertyGroup^>>> %PROJECT_FILE%
@ECHO     ^<OutDir^>./^</OutDir^>>> %PROJECT_FILE%
@ECHO     ^<IntDir^>$(Platform)\$(Configuration)\^</IntDir^>>> %PROJECT_FILE%
@ECHO     ^<TargetName^>$(ProjectName)_$(Configuration)^</TargetName^>>> %PROJECT_FILE%
@ECHO   ^</PropertyGroup^>>> %PROJECT_FILE%
@ECHO   ^<ItemDefinitionGroup^>>> %PROJECT_FILE%
@ECHO     ^<ClCompile^>>> %PROJECT_FILE%
@ECHO       ^<WarningLevel^>Level3^</WarningLevel^>>> %PROJECT_FILE%
@ECHO       ^<Optimization Condition="'$(Configuration)'=='Release'"^>MaxSpeed^</Optimization^>>> %PROJECT_FILE%
@ECHO       ^<Optimization^>Disabled^</Optimization^>>> %PROJECT_FILE%
@ECHO       ^<FunctionLevelLinking Condition="'$(Configuration)'=='Release'"^>true^</FunctionLevelLinking^>>> %PROJECT_FILE%
@ECHO       ^<IntrinsicFunctions Condition="'$(Configuration)'=='Release'"^>true^</IntrinsicFunctions^>>> %PROJECT_FILE%
@ECHO       ^<SDLCheck^>true^</SDLCheck^>>> %PROJECT_FILE%
@ECHO       ^<AdditionalIncludeDirectories^>%CODE_DIRECTORY%%INCLUDE_DIRS%^</AdditionalIncludeDirectories^>>> %PROJECT_FILE%
@ECHO     ^</ClCompile^>>> %PROJECT_FILE%
@ECHO     ^<Link^>>> %PROJECT_FILE%
@ECHO       ^<GenerateDebugInformation^>true^</GenerateDebugInformation^>>> %PROJECT_FILE%
@ECHO       ^<EnableCOMDATFolding Condition="'$(Configuration)'=='Release'"^>true^</EnableCOMDATFolding^>>> %PROJECT_FILE%
@ECHO       ^<OptimizeReferences Condition="'$(Configuration)'=='Release'"^>true^</OptimizeReferences^>>> %PROJECT_FILE%
@ECHO       ^<AdditionalDependencies Condition="'$(Configuration)'=='Release'"^>runner_Release.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^>>> %PROJECT_FILE%
@ECHO       ^<AdditionalDependencies Condition="'$(Configuration)'=='Debug'"^>runner_Debug.lib^;%%(AdditionalDependencies)^</AdditionalDependencies^>>> %PROJECT_FILE%
@ECHO     ^</Link^>>> %PROJECT_FILE%
@ECHO   ^</ItemDefinitionGroup^>>> %PROJECT_FILE%
@ECHO   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" /^>>> %PROJECT_FILE%
@ECHO   ^<ImportGroup Label="ExtensionTargets"^>>> %PROJECT_FILE%
@ECHO   ^</ImportGroup^>>> %PROJECT_FILE%
IF DEFINED INCLUDE_USERPROJECT (
    @ECHO   ^<Target Name="FindUserProjects" BeforeTargets="PrepareForBuild"^>>> %PROJECT_FILE%
    @ECHO     ^<Exec Command="where genn-buildmodel.bat" ConsoleToMsBuild="true"^>>> %PROJECT_FILE%
    @ECHO       ^<Output TaskParameter="ConsoleOutput" PropertyName="GeNNBuildModelPath" /^>>> %PROJECT_FILE%
    @ECHO     ^</Exec^>>> %PROJECT_FILE%
    @ECHO     ^<ItemGroup^>>> %PROJECT_FILE%
	@ECHO       ^<ClCompile^>>> %PROJECT_FILE%
	 @ECHO        ^<AdditionalIncludeDirectories^>%%^(AdditionalIncludeDirectories^)^;$^([System.IO.Path]::GetFullPath^($([System.IO.Path]::GetDirectoryName^($^(GeNNBuildModelPath^)^)^)\..\userproject\include^)^)^</AdditionalIncludeDirectories^>>> %PROJECT_FILE%
	@ECHO       ^</ClCompile^>>> %PROJECT_FILE%
	@ECHO     ^</ItemGroup^>>> %PROJECT_FILE%
    @ECHO   ^</Target^>>> %PROJECT_FILE%
)
@ECHO ^</Project^>>> %PROJECT_FILE%

REM Write out MSBuild solution

@ECHO Microsoft Visual Studio Solution File, Format Version 12.00> %SOLUTION_FILE%
@ECHO # Visual Studio 2013>> %SOLUTION_FILE%
@ECHO VisualStudioVersion = 12.0.30501.0>> %SOLUTION_FILE%
@ECHO MinimumVisualStudioVersion = 10.0.40219.1>> %SOLUTION_FILE%
@ECHO Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "%PROJECT_NAME%", "%PROJECT_FILE%", "{%USER_GUID%}">> %SOLUTION_FILE%
@ECHO 	ProjectSection(ProjectDependencies) = postProject>> %SOLUTION_FILE%
@ECHO 		{%RUNNER_GUID%} = {%RUNNER_GUID%}>> %SOLUTION_FILE%
@ECHO 	EndProjectSection>> %SOLUTION_FILE%
@ECHO EndProject>> %SOLUTION_FILE%
@ECHO Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "runner", "%CODE_DIRECTORY%\runner.vcxproj", "{%RUNNER_GUID%}">> %SOLUTION_FILE%
@ECHO EndProject>> %SOLUTION_FILE%
@ECHO Global>> %SOLUTION_FILE%
@ECHO 	GlobalSection(SolutionConfigurationPlatforms) = preSolution>> %SOLUTION_FILE%
@ECHO 		Debug^|x64 = Debug^|x64>> %SOLUTION_FILE%
@ECHO 		Release^|x64 = Release^|x64>> %SOLUTION_FILE%
@ECHO 	EndGlobalSection>> %SOLUTION_FILE%
@ECHO 	GlobalSection(ProjectConfigurationPlatforms) = postSolution>> %SOLUTION_FILE%
@ECHO 		{%USER_GUID%}.Debug^|x64.ActiveCfg = Debug^|x64>> %SOLUTION_FILE%
@ECHO 		{%USER_GUID%}.Debug^|x64.Build.0 = Debug^|x64>> %SOLUTION_FILE%
@ECHO 		{%USER_GUID%}.Release^|x64.ActiveCfg = Release^|x64>> %SOLUTION_FILE%
@ECHO 		{%USER_GUID%}.Release^|x64.Build.0 = Release^|x64>> %SOLUTION_FILE%
@ECHO 		{%RUNNER_GUID%}.Debug^|x64.ActiveCfg = Debug^|x64>> %SOLUTION_FILE%
@ECHO 		{%RUNNER_GUID%}.Debug^|x64.Build.0 = Debug^|x64>> %SOLUTION_FILE%
@ECHO 		{%RUNNER_GUID%}.Release^|x64.ActiveCfg = Release^|x64>> %SOLUTION_FILE%
@ECHO 		{%RUNNER_GUID%}.Release^|x64.Build.0 = Release^|x64>> %SOLUTION_FILE%
@ECHO 	EndGlobalSection>> %SOLUTION_FILE%
@ECHO 	GlobalSection(SolutionProperties) = preSolution>> %SOLUTION_FILE%
@ECHO 		HideSolutionNode = FALSE>> %SOLUTION_FILE%
@ECHO 	EndGlobalSection>> %SOLUTION_FILE%
@ECHO EndGlobal>> %SOLUTION_FILE%
