#include "code_generator/generateMSBuild.h"

// Standard C++ includes
#include <string>

// GeNN code generator includes
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateMSBuild(std::ostream &os, const BackendBase &backend, const std::string &projectGUID,
    const std::vector<std::string> &moduleNames)
{
    // Generate header and targets for release and debug builds
    os << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
    os << "<Project DefaultTargets=\"Build\" ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">" << std::endl;
    os << "\t<ItemGroup Label=\"ProjectConfigurations\">" << std::endl;
    os << "\t\t<ProjectConfiguration Include=\"Debug|x64\">" << std::endl;
    os << "\t\t\t<Configuration>Debug</Configuration>" << std::endl;
    os << "\t\t\t<Platform>x64</Platform>" << std::endl;
    os << "\t\t</ProjectConfiguration>" << std::endl;
    os << "\t\t<ProjectConfiguration Include=\"Release|x64\">" << std::endl;
    os << "\t\t\t<Configuration>Release</Configuration>" << std::endl;
    os << "\t\t\t<Platform>x64</Platform>" << std::endl;
    os << "\t\t</ProjectConfiguration>" << std::endl;
    os << "\t</ItemGroup>" << std::endl;

    os << "\t<PropertyGroup Label=\"Globals\">" << std::endl;
    if (!projectGUID.empty()) {
        os << "\t\t<ProjectGuid>{" << projectGUID << "}</ProjectGuid>" << std::endl;
    }
    os << "\t\t<RootNamespace>runner</RootNamespace>" << std::endl;
    os << "\t</PropertyGroup>" << std::endl;

    // Import C++ default props
    os << "\t<Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.Default.props\" />" << std::endl;
    
    // Generate property group containing configuration properties
    os << "\t<PropertyGroup Label=\"Configuration\">" << std::endl;
    os << "\t\t<ConfigurationType>DynamicLibrary</ConfigurationType>" << std::endl;
    os << "\t\t<UseDebugLibraries Condition=\"'$(Configuration)'=='Release'\">false</UseDebugLibraries>" << std::endl;
    os << "\t\t<UseDebugLibraries Condition=\"'$(Configuration)'=='Debug'\">true</UseDebugLibraries>" << std::endl;
    os << "\t\t<CharacterSet>MultiByte</CharacterSet>" << std::endl;
    os << "\t\t<PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>" << std::endl;
    os << "\t\t<PreferredToolArchitecture>x64</PreferredToolArchitecture>" << std::endl;
    backend.genMSBuildConfigProperties(os);
    os << "\t</PropertyGroup>" << std::endl;
    
    // Import C++ props
    os << "\t<Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.props\" />" << std::endl;
    backend.genMSBuildImportProps(os);

    os << "\t<ImportGroup Label=\"PropertySheets\">" << std::endl;
    os << "\t\t<Import Project=\"$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\" Condition=\"exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />" << std::endl;
    os << "\t</ImportGroup>" << std::endl;

    // Generate property group configuring build target
    os << "\t<PropertyGroup>" << std::endl;
    os << "\t\t<LinkIncremental Condition=\"'$(Configuration)'=='Debug'\">true</LinkIncremental>" << std::endl;
    os << "\t\t<OutDir>../</OutDir>" << std::endl;
    os << "\t\t<TargetName>$(ProjectName)_$(Configuration)</TargetName>" << std::endl;
    os << "\t\t<TargetExt>.dll</TargetExt>" << std::endl;
    os << "\t</PropertyGroup>" << std::endl;

    // Generate item definitions, configuring how items are built
    os << "\t<ItemDefinitionGroup>" << std::endl;
    backend.genMSBuildItemDefinitions(os);
    os << "\t</ItemDefinitionGroup>" << std::endl;

    // Generate itemgroup to compile modules
    os << "\t<ItemGroup>" << std::endl;
    for(const auto &m : moduleNames) {
        backend.genMSBuildCompileModule(m, os);
    }
    os << "\t</ItemGroup>" << std::endl;
     
    // Generate postamble, importing targets
    os << "<Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.targets\" />" << std::endl;
    backend.genMSBuildImportTarget(os);
    os << "</Project>" << std::endl;
}