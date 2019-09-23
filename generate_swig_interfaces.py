"""Generate SWIG interfaces
This module generates a number of SWIG interface (.i) files and Custom model
header and source files.

Generated interface files are:

    -- genn_weapper.i - interface of the main module
    -- NeuronModels.i -- interface of the NeuronModels module
    -- PostsynapticModels.i -- interface of the PostsynapticModels module
    -- WeightUpdateModels.i -- interface of the WeightUpdateModels module
    -- currentSourceModels.i -- interface of the CurrentSourceModels module
    -- initVarSnippet.i -- interface of the InitVarSnippet module
    -- stl_containers.i -- interface of the StlContainers module which wraps
                           different specialization of std::vector and std::pair
    -- SharedLibraryModel.i -- interface of the SharedLibraryModel module which
                is used to load model at runtime

Generated headers and sources are:

    -- newNeuronModelsCustom.h/.cc -- header and source files for NeuronModels::Custom class
    -- newWeightUpdateModelsCustom.h/.cc -- header and source files for WeightUpdateModels::Custom class
    -- newPostsynapticModelsCustom.h/.cc -- header and source files for PostsynapticModels::Custom class
    -- currentSourceModelsCustom.h/.cc -- header and source files for CurrentSourceModels::Custom class
    -- initVarSnippetCustom.h/.cc -- header and source files for InitVarSnippet::Custom class

Example:
    $ python generate_swig_interfaces.py path_to_pygenn

Attrbutes:

    NEURONMODELS -- common name of NeuronModels header and interface files without extention
    POSTSYNMODELS -- common name of PostsynapticModels header and interface files without extention
    WUPDATEMODELS -- common name of WeightUpdateModels header and interface files without extention
    CURRSOURCEMODELS -- common name of CurrentSourceModels header and interface files without extention
    INITVARSNIPPET -- common name of InitVarSnippet header and interface files without extention
    MAIN_MODULE -- name of the main SWIG module
"""
import os  # to work with paths nicely
from itertools import product
from string import Template # for better text substitutions
from argparse import ArgumentParser # to parse command line args

# module attributes
NEURONMODELS = 'neuronModels'
POSTSYNMODELS = 'postsynapticModels'
WUPDATEMODELS = 'weightUpdateModels'
CURRSOURCEMODELS = 'currentSourceModels'
INITVARSNIPPET = 'initVarSnippet'
SPARSEINITSNIPPET = 'initSparseConnectivitySnippet'
MAIN_MODULE = 'genn_wrapper'

# Scope classes should be used with 'with' statement. They write code in the
# beginning and in the end of the with-block.
class SwigInlineScope( object ):
    def __init__( self, ofs ):
        '''Adds %inline block. The code within %inline %{ %} block is added to the generated wrapper C++ file AND is processed by SWIG'''
        self.os = ofs
    def __enter__( self ):
        self.os.write( '\n%inline %{\n' )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '%}\n' )

class SwigExtendScope( object ):
    def __init__( self, ofs, classToExtend ):
        '''Adds %extend block. The code within %extend classToExtend { } block is used to add functionality to classes without touching the original implementation'''
        self.os = ofs
        self.classToExtend = classToExtend
    def __enter__( self ):
        self.os.write( '\n%extend ' + self.classToExtend + ' {\n' )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '};\n' )

class SwigAsIsScope( object ):
    def __init__( self, ofs ):
        '''Adds %{ %} block. The code within %{ %} block is added to the wrapper C++ file as is without being processed by SWIG'''
        self.os = ofs
    def __enter__( self ):
        self.os.write( '\n%{\n' )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '%}\n' )

class SwigInitScope( object ):
    def __init__( self, ofs ):
        '''Adds %init %{ %} block. The code within %{ %} block is copied directly into the module initialization function'''
        self.os = ofs
    def __enter__( self ):
        self.os.write( '\n%init %{\n' )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '%}\n' )

class CppBlockScope( object ):
    def __init__( self, ofs ):
        '''Adds a C-style block'''
        self.os = ofs
    def __enter__( self ):
        self.os.write( '\n{\n' )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '}\n' )

class SwigModuleGenerator( object ):

    '''A helper class for generating SWIG interface files'''

    def __init__( self, moduleName, outFile ):
        '''Init SwigModuleGenerator

        Arguments:
            moduleName -- string, name of the SWIG module
            outFile -- string, output file
        '''
        self.name = moduleName
        self.outFile = outFile

    # __enter__ and __exit__ are functions which are called if the class is created
    # using 'with' statement. __enter__ is called in the very beginning, and
    # __exit__ is called when the indented with-block ends.
    def __enter__(self):
        self.os = open( self.outFile, 'w' )
        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.close()

    def addSwigModuleHeadline( self, directors=False, comment='' ):
        '''Adds a line naming a module and enabling directors feature for inheritance in python (disabled by default)'''
        optionsCode = ''
        if directors:
            optionsCode = '(package="%s", directors="1")' % MAIN_MODULE
        else:
            optionsCode = '(package="%s")' % MAIN_MODULE
        self.write( '%module{} {} {}\n'.format( optionsCode, self.name, comment ) )

    def addSwigFeatureDirector( self, cClassName, comment='' ):
        '''Adds a line enabling director feature for a C/C++ class'''
        self.write( '%feature("director") {}; {}\n'.format( cClassName, comment ) )

    def addSwigInclude( self, cHeader, comment='' ):
        '''Adds a line for including C/C++ header file. %include statement makes SWIG to include the header into generated C/C++ file code AND to process it'''
        self.write( '%include {} {}\n'.format( cHeader, comment ) )

    def addSwigImport( self, swigIFace, comment='' ):
        '''Adds a line for importing SWIG interface file. %import statement notifies SWIG about class(es) covered in another module'''
        self.write( '%import {} {}\n'.format( swigIFace, comment ) )

    def addSwigIgnore( self, identifier, comment='' ):
        '''Adds a line instructing SWIG to ignore the matching identifier'''
        self.write( '%ignore {}; {}\n'.format( identifier, comment ) )

    def addSwigRename( self, identifier, newName, comment='' ):
        '''Adds a line instructing SWIG to rename the matching identifier'''
        self.write( '%rename({}) {}; {}\n'.format( newName, identifier, comment ) )

    def addSwigUnignore( self, identifier ):
        '''Adds a line instructing SWIG to unignore the matching identifier'''
        self.addSwigRename( identifier, '"%s"', '// unignore' )

    def addSwigIgnoreAll( self ):
        '''Adds a line instructing SWIG to ignore everything, but templates. Unignoring templates does not work well in SWIG. Can be used for fine-grained control over what is wrapped'''
        self.addSwigRename( '""', '"$ignore"', '// ignore all' )

    def addSwigUnignoreAll( self ):
        '''Adds a line instructing SWIG to unignore everything'''
        self.addSwigRename( '""', '"%s"', '// unignore all' )

    def addSwigEnableUnderCaseConvert( self ):
        self.addSwigRename('""', '"%(undercase)s", %$isfunction, notregexmatch$name="add[a-zA-Z]*Population", notregexmatch$name="addCurrentSource", notregexmatch$name="assignExternalPointer[a-zA-Z]*"', '// Enable conversion to under_case')

    def addSwigTemplate( self, tSpec, newName ):
        '''Adds a template specification tSpec and renames it as newName'''
        self.write( '%template({}) {};\n'.format( newName, tSpec ) )

    def addCppInclude( self, cHeader, comment='' ):
        '''Adds a line for usual including C/C++ header file.'''
        self.write( '#include {} {}\n'.format( cHeader, comment ) )

    def addAutoGenWarning( self ):
        '''Adds a comment line telling that this file was generated automatically'''
        self.write( '// This code was generated by ' + os.path.split(__file__)[1] + '. DO NOT EDIT\n' )

    def write( self, code ):
        '''Writes code into output stream os'''
        self.os.write( code )


def writeValueMakerFunc( modelName, valueName, numValues, mg ):
    '''Generates a helper make*Values function and writes it'''

    vals = 'vals'
    if numValues == 0:
        vals = ''
    paramType = 'double'
    if 'VarValues' in valueName:
        paramType = 'Models::VarInit'

    mg.write( 'static {0}::{1}::{2}* make{2}( const std::vector<{3}> & {4} )'.format(
        mg.name,
        modelName,
        valueName,
        paramType,
        vals
        ) )
    with CppBlockScope( mg ):
        if numValues != 0:
            mg.write( paramType + ' ' + ', '.join( ['v{0} = vals[{0}]'.format( i )
                                          for i in range(numValues)] ) + ';\n' )
        mg.write( 'return new {}::{}({});\n'.format(
            mg.name + '::' + modelName,
            valueName,
            ', '.join( [('v' + str( i )) for i in range(numValues)] ) ) )


def generateCustomClassDeclaration( nSpace, initVarSnippet=False, weightUpdateModel=False ):
    '''Generates nSpace::Custom class declaration string'''

    varValuesTypedef = ''
    varValuesMaker = ''
    if not initVarSnippet:
        varValuesTypedef = 'typedef CustomValues::VarValues VarValues;'
        varValuesMaker = '''static CustomValues::VarValues* makeVarValues( const std::vector< Models::VarInit > & vals )
        {
            return new CustomValues::VarValues( vals );
        }'''
        
    if weightUpdateModel:
        varValuesTypedef += '''
        typedef CustomValues::VarValues PreVarValues;
        typedef CustomValues::VarValues PostVarValues;'''

        varValuesMaker += '''
        static CustomValues::VarValues* makePreVarValues( const std::vector< Models::VarInit > & vals )
        {
            return new CustomValues::VarValues( vals );
        }
        static CustomValues::VarValues* makePostVarValues( const std::vector< Models::VarInit > & vals )
        {
            return new CustomValues::VarValues( vals );
        }'''
    
    return Template('''
namespace ${NAMESPACE}
{
class Custom : public Base
{
public:
    typedef CustomValues::ParamValues ParamValues;
    ${varValuesTypedef}
    static CustomValues::ParamValues* makeParamValues( const std::vector< double > & vals )
    {
        return new CustomValues::ParamValues( vals );
    }
    ${varValuesMaker}
};
} // namespace ${NAMESPACE}
''').substitute(NAMESPACE=nSpace, varValuesTypedef=varValuesTypedef, varValuesMaker=varValuesMaker)

def generateNumpyApplyArgoutviewArray1D( dataType, varName, sizeName ):
    '''Generates a line which applies numpy ARGOUTVIEW_ARRAY1 typemap to variable. ARGOUTVIEW_ARRAY1 gives access to C array via numpy array.'''
    return Template( '%apply ( ${data_t}* ARGOUTVIEW_ARRAY1, int* DIM1 ) {( ${data_t}* ${varName}, int* ${sizeName} )};\n').substitute( data_t=dataType, varName=varName, sizeName=sizeName )

def generateNumpyApplyInArray1D( dataType, varName, sizeName ):
    '''Generates a line which applies numpy IN_ARRAY1 typemap to variable. IN_ARRAY1 is used to pass a numpy array as C array to C code'''
    return Template( '%apply ( ${data_t} IN_ARRAY1, int DIM1 ) {( ${data_t} ${varName}, int ${sizeName} )};\n').substitute( data_t=dataType, varName=varName, sizeName=sizeName )

def generateBuiltInGetter( models ):
    return Template('''std::vector< std::string > getBuiltInModels() {
    return std::vector<std::string>{"${MODELS}"};
}
''').substitute( MODELS='", "'.join( models ) )


def generateSharedLibraryModelInterface( swigPath ):
    '''Generates SharedLibraryModelNumpy.i file'''
    with SwigModuleGenerator('SharedLibraryModelNumpy',
            os.path.join( swigPath, 'SharedLibraryModelNumpy.i' ) ) as mg:
        mg.addAutoGenWarning()
        mg.addSwigModuleHeadline()
        with SwigAsIsScope( mg ):
            mg.write( '#define SWIG_FILE_WITH_INIT // for numpy\n' )
            mg.addCppInclude( '"sharedLibraryModelNumpy.h"' )

        mg.addSwigInclude( '<std_string.i>' )
        mg.addSwigInclude( '"numpy.i"' )
        with SwigInitScope( mg ):
            mg.write( 'import_array();\n')

        # These are all data types supported by numpy SWIG interface (at least by default) plus long double
        npDTypes = ('signed char', 'unsigned char',
                    'short', 'unsigned short',
                    'int', 'unsigned int',
                    'long', 'unsigned long',
                    'long long', 'unsigned long long',
                    'float', 'double')

        for dataType in [dt+'*' for dt in npDTypes]:
            mg.write( generateNumpyApplyArgoutviewArray1D( dataType, 'varPtr', 'n1' ) )
        mg.write( generateNumpyApplyInArray1D( 'double*', '_g', 'nG' ) )
        mg.write( generateNumpyApplyInArray1D( 'float*', '_g', 'nG' ) )

        mg.addSwigEnableUnderCaseConvert()
        mg.addSwigInclude( '"../../../userproject/include/sharedLibraryModel.h"' )
        mg.addSwigInclude( '"sharedLibraryModelNumpy.h"' )
        for dtShort, dataType in zip( [ "".join([dt_[0] for dt_ in dt.split()]) for dt in npDTypes],
                npDTypes ):
            mg.addSwigTemplate( 'SharedLibraryModelNumpy::assignExternalPointerArray<{}>'.format( dataType ),
                'assign_external_pointer_array_' + dtShort )
            mg.addSwigTemplate( 'SharedLibraryModelNumpy::assignExternalPointerSingle<{}>'.format( dataType ),
                'assign_external_pointer_single_' + dtShort )

        for dtShort, dataType in zip(('f', 'd'), ('float', 'double')):
            mg.addSwigTemplate( 'SharedLibraryModel<{}>'.format( dataType ),
                'SharedLibraryModel_' + dtShort )
            mg.addSwigTemplate( 'SharedLibraryModelNumpy<{}>'.format( dataType ),
                'SharedLibraryModelNumpy_' + dtShort )


def generateStlContainersInterface( swigPath ):
    '''Generates StlContainers interface which wraps std::string, std::pair,
       std::vector, std::function and creates template specializations for pairs and vectors'''
    with SwigModuleGenerator( 'StlContainers',
            os.path.join( swigPath, 'StlContainers.i' ) ) as mg:
        mg.addAutoGenWarning()
        mg.addSwigModuleHeadline()
        with SwigAsIsScope( mg ):
            mg.addCppInclude( '<functional>', '// for std::function' )


        mg.write( '\n// swig wrappers for STL containers\n' )
        mg.addSwigInclude( '<std_string.i>' )
        mg.addSwigInclude( '<std_pair.i>' )
        mg.addSwigInclude( '<std_vector.i>' )

        dpfunc_template_spec = 'function<double( const std::vector<double> &, double )>'
        mg.write( '\n// wrap std::function in a callable struct with the same name\n' )
        mg.write( '// and enable directors feature for it, so that a new class can\n' )
        mg.write( '// be derived from it in python. swig magic.\n')
        mg.addSwigRename( 'std::' + dpfunc_template_spec,
                'STD_DPFunc' )
        mg.addSwigRename(
                'std::' + dpfunc_template_spec + '::operator()',
                '__call__',
                '// rename operator() as __call__ so that it works correctly in python' )
        mg.addSwigFeatureDirector( 'std::' + dpfunc_template_spec )
        mg.write('namespace std{\n'
                 '    struct ' + dpfunc_template_spec + ' {\n'
                 '        // copy ctor\n' +
                 '        {0}( const std::{0}&);\n'.format( dpfunc_template_spec ) +
                 '        double operator()( const std::vector<double> &, double ) const;\n'
                 '    };\n'
                 '}\n' )

        mg.write( '\n// add template specifications for various STL containers\n' )
        mg.addSwigTemplate( 'std::vector<std::string>', 'StringVector' )

        # These are all data types supported by numpy SWIG interface (at least by default) plus long double
        npDTypes = (
                'signed char', 'unsigned char',
                'short', 'unsigned short',
                'int', 'unsigned int',
                'long', 'unsigned long',
                'long long', 'unsigned long long',
                'float', 'double', 'long double'
        )

        for npDType in npDTypes:
            camelDT = ''.join( x.capitalize() for x in npDType.split() ) + 'Vector'
            mg.addSwigTemplate( 'std::vector<{}>'.format(npDType), camelDT )


def generateCustomModelDeclImpls(swigPath):
    '''Generates headers/sources with *::Custom classes'''
    models = [NEURONMODELS, POSTSYNMODELS, WUPDATEMODELS, CURRSOURCEMODELS, INITVARSNIPPET, SPARSEINITSNIPPET]
    for model in models:
        nSpace = model[0].upper() + model[1:]
        with SwigModuleGenerator( 'decl',
                os.path.join( swigPath, model + 'Custom.h' ) ) as mg:
            mg.addAutoGenWarning()
            mg.write( '#pragma once\n' )
            mg.addCppInclude( '"' + model + '.h"' )
            mg.addCppInclude( '"customParamValues.h"' )
            if model != INITVARSNIPPET and model != SPARSEINITSNIPPET:
                mg.addCppInclude( '"customVarValues.h"' )
            mg.write(generateCustomClassDeclaration(nSpace, model==INITVARSNIPPET or model==SPARSEINITSNIPPET, model==WUPDATEMODELS))
        with SwigModuleGenerator( 'impl',
                os.path.join( swigPath, model + 'Custom.cc' ) ) as mg:
            mg.addAutoGenWarning()
            mg.addCppInclude( '"' + model + 'Custom.h"' )

def generateBackend(swigPath, folder, namespace):
    # Create SWIG module
    with SwigModuleGenerator(namespace + "Backend", os.path.join(swigPath, namespace + "Backend.i")) as mg:
        mg.addAutoGenWarning()
        mg.addSwigModuleHeadline()
        mg.addSwigEnableUnderCaseConvert()
        mg.addSwigInclude('<exception.i>')
        
        with SwigAsIsScope(mg):
            mg.addCppInclude('"optimiser.h"')
            mg.addCppInclude('"path.h"')
            mg.write("using namespace CodeGenerator::" + namespace + ";\n")

        # Include genn and backend export headers so BACKEND_EXPORT and GENN_EXPORT macros can be correctly parsed
        mg.addSwigInclude( '"backendExport.h"' )
        mg.addSwigInclude( '"gennExport.h"' )
        
        # Parse backend base, ignore BackendBase itself to get PreferencesBase definition
        mg.addSwigIgnore("BackendBase")
        mg.addSwigInclude('"code_generator/backendBase.h"')

        # Parse backend, ignore Backend itself to get PreferencesBase definition
        mg.addSwigIgnore("Backend")
        mg.addSwigInclude('"backend.h"')

        # Import stl containers so as to support std::string
        mg.addSwigImport( '"StlContainers.i"' )

        # Include SWIG exception handling library
        mg.addSwigInclude('<exception.i>')
        
        mg.write('''
        %exception create_backend{
            try 
            {
                $action
            } 
            SWIG_CATCH_STDEXCEPT
            catch (...) {
                SWIG_exception(SWIG_UnknownError, "Unknown exception");
            }
        }
        ''')
        
        # To prevent having to expose filesystem, simply export a wrapper that converts a string to a filesystem::path and calls createBackend
        with SwigInlineScope(mg):
            mg.write('CodeGenerator::' + namespace + '::Backend create_backend(const ModelSpecInternal &model, const std::string &outputPath, int localHostID, const CodeGenerator::' + namespace + '::Preferences &preferences)\n'
                     '{\n'
                     '  return Optimiser::createBackend(model, filesystem::path(outputPath), localHostID, preferences);\n'
                     '}\n\n'
                     'void delete_backend(CodeGenerator::' + namespace + '::Backend *backend)\n'
                     '{\n'
                     '  delete backend;\n'
                     '}\n')


def generateConfigs(gennPath, backends):
    swigPath = os.path.join(gennPath, 'pygenn', 'genn_wrapper', 'generated')
    includePath = os.path.join(gennPath, 'include', 'genn', 'genn')
    
    # Create output path if it doesn't exist
    if not os.path.exists(swigPath):
        os.makedirs(swigPath)

    # Generates SWIG interfaces
    generateStlContainersInterface(swigPath)
    generateCustomModelDeclImpls(swigPath)
    generateSharedLibraryModelInterface(swigPath)
    
    # Generate SWIG interfaces for all supported backends
    for (f, n, _) in backends:
        generateBackend(swigPath, f, n)

    # open header files with models and instantiate SwigModuleGenerators
    with open( os.path.join( includePath, NEURONMODELS + ".h" ), 'r' ) as neuronModels_h, \
            open( os.path.join( includePath, POSTSYNMODELS + ".h" ), 'r' ) as postsynModels_h, \
            open( os.path.join( includePath, WUPDATEMODELS + ".h" ), 'r' ) as wUpdateModels_h, \
            open( os.path.join( includePath, CURRSOURCEMODELS + ".h" ), 'r' ) as currSrcModels_h, \
            open( os.path.join( includePath, INITVARSNIPPET + ".h" ), 'r' ) as initVarSnippet_h, \
            open( os.path.join( includePath, SPARSEINITSNIPPET + ".h" ), 'r' ) as sparseInitSnippet_h, \
            SwigModuleGenerator( MAIN_MODULE, os.path.join( swigPath, MAIN_MODULE + '.i' ) ) as pygennSmg, \
            SwigModuleGenerator( 'NeuronModels', os.path.join( swigPath, 'NeuronModels.i' ) ) as neuronSmg, \
            SwigModuleGenerator( 'PostsynapticModels', os.path.join( swigPath, 'PostsynapticModels.i' ) ) as postsynSmg, \
            SwigModuleGenerator( 'WeightUpdateModels', os.path.join( swigPath, 'WeightUpdateModels.i' ) ) as wUpdateSmg, \
            SwigModuleGenerator( 'CurrentSourceModels', os.path.join( swigPath, 'CurrentSourceModels.i' ) ) as currSrcSmg, \
            SwigModuleGenerator( 'InitVarSnippet', os.path.join( swigPath, 'InitVarSnippet.i' ) ) as iniVarSmg, \
            SwigModuleGenerator( 'InitSparseConnectivitySnippet', os.path.join( swigPath, 'InitSparseConnectivitySnippet.i' ) ) as iniSparseSmg:

        # pygennSmg generates main SWIG interface file,
        # mgs generate SWIG interfaces for models and InitVarSnippet

        mgs = [ neuronSmg, postsynSmg, wUpdateSmg, currSrcSmg, iniVarSmg, iniSparseSmg]

        pygennSmg.addAutoGenWarning()
        pygennSmg.addSwigModuleHeadline()
        with SwigAsIsScope( pygennSmg ):
            pygennSmg.addCppInclude( '<fstream>' )
            pygennSmg.addCppInclude( '"variableMode.h"' )
            pygennSmg.addCppInclude( '"modelSpec.h"' )
            pygennSmg.addCppInclude( '"modelSpecInternal.h"' )
            pygennSmg.addCppInclude( '"synapseMatrixType.h"' )
            pygennSmg.addCppInclude( '"neuronGroup.h"' )
            pygennSmg.addCppInclude( '"synapseGroup.h"' )
            pygennSmg.addCppInclude( '"currentSource.h"' )
            pygennSmg.addCppInclude( '"modelSpec.h"' )
            for header in (NEURONMODELS, POSTSYNMODELS,
                           WUPDATEMODELS, CURRSOURCEMODELS, INITVARSNIPPET, SPARSEINITSNIPPET):
                pygennSmg.addCppInclude( '"' + header + 'Custom.h"' )
            pygennSmg.addCppInclude( '"code_generator/generateAll.h"' )
            pygennSmg.addCppInclude( '"code_generator/generateMakefile.h"' )
            pygennSmg.addCppInclude( '"code_generator/generateMSBuild.h"' )
            pygennSmg.addCppInclude( '"path.h"' )
        pygennSmg.addSwigImport( '"StlContainers.i"' )
        
        # Include genn export header so GENN_EXPORT macros can be correctly parsed
        pygennSmg.addSwigInclude( '"gennExport.h"' )
        
        # define and wrap two functions which replace main in generateALL.cc
        with SwigInlineScope( pygennSmg ):
            pygennSmg.write( '''
            void generate_code(ModelSpecInternal &model, CodeGenerator::BackendBase &backend, const std::string &path, int localHostID) {
                const filesystem::path outputPath(path);

                // Generate code, returning list of module names that must be build
                const auto moduleNames = CodeGenerator::generateAll(model, backend, outputPath);

            #ifdef _WIN32
                // Create MSBuild project to compile and link all generated modules
                std::ofstream makefile((outputPath / "runner.vcxproj").str());
                CodeGenerator::generateMSBuild(makefile, backend, "", moduleNames);
            #else
                // Create makefile to compile and link all generated modules
                std::ofstream makefile((outputPath / "Makefile").str());
                CodeGenerator::generateMakefile(makefile, backend, moduleNames);
            #endif
            }
            ''' )

        # generate SWIG interface files for models and InitVarSnippet
        for mg, header in zip(mgs, (neuronModels_h, postsynModels_h,
                                    wUpdateModels_h, currSrcModels_h,
                                    initVarSnippet_h, sparseInitSnippet_h)):
            _, headerFilename = os.path.split( header.name )
            is_snippet = (mg.name == 'InitVarSnippet' or mg.name == 'InitSparseConnectivitySnippet')

            pygennSmg.addSwigImport( '"' + mg.name + '.i"' )
            mg.addAutoGenWarning()
            mg.addSwigModuleHeadline( directors = True )
            with SwigAsIsScope( mg ):
                mg.addCppInclude( '"' + headerFilename + '"' )
                mg.addCppInclude( '"' + headerFilename.split('.')[0] + 'Custom.h"' )
                mg.addCppInclude( '"customParamValues.h"' )
                if not is_snippet:
                    mg.addCppInclude( '"initVarSnippetCustom.h"' )
                    mg.addCppInclude( '"customVarValues.h"' )
            
            # Include genn export header so GENN_EXPORT macros can be correctly parsed
            mg.addSwigInclude( '"gennExport.h"' )
            
            if is_snippet:
                mg.addSwigImport( '"Snippet.i"' )
            else:
                mg.addSwigImport( '"Models.i"' )
            mg.addSwigFeatureDirector( mg.name + '::Base' )
            mg.addSwigInclude( '"' + headerFilename + '"' )
            mg.addSwigFeatureDirector( mg.name + '::Custom' )
            mg.addSwigInclude( '"' + headerFilename.split('.')[0] + 'Custom.h"' )

            mg.models = []

            # parse models files and collect models declared there
            for line in header.readlines():
                line = line.lstrip()
                line = line.rstrip()
                if line.startswith( 'DECLARE_' ) and line.endswith(';'):
                    is_new_wum_declaration = mg.name == 'WeightUpdateModels' and line.startswith('DECLARE_WEIGHT_UPDATE_MODEL')
                    if is_snippet:
                        nspace_model_name, num_params = line.split( '(' )[1].split( ')' )[0].split( ',' )
                    elif is_new_wum_declaration:
                        nspace_model_name, num_params, num_vars, num_pre_vars, num_post_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )
                    else:
                        nspace_model_name, num_params, num_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )

                    nspace_model_name_comps = nspace_model_name.split( '::' )
                    model_name = nspace_model_name_comps[-1]

                    mg.models.append( model_name )

                    # add a helper function to create Param- and VarVarlues to each model
                    with SwigExtendScope( mg, mg.name + '::' + model_name ):
                        writeValueMakerFunc( model_name, 'ParamValues', int(num_params), mg )
                        if not is_snippet:
                            writeValueMakerFunc( model_name, 'VarValues', int(num_vars), mg )

                        if is_new_wum_declaration:
                            writeValueMakerFunc( model_name, 'PreVarValues', int(num_pre_vars), mg )
                            writeValueMakerFunc( model_name, 'PostVarValues', int(num_post_vars), mg )

        # Add wrapper around InitSparseConnectivitySnippet::Base::CalcMaxLengthFunc
        iniSparseSmg.write('''
            %feature("director") CalcMaxLengthFunc;
            %rename(__call__) CalcMaxLengthFunc::operator();
            %inline %{
            struct CalcMaxLengthFunc {
            virtual unsigned int operator()( unsigned int, unsigned int, const std::vector<double> & pars ) const = 0;
            virtual ~CalcMaxLengthFunc() {}
            };
            %}

            // helper function to convert CalcMaxLengthFunc to std::function
            %inline %{
            std::function<unsigned int( unsigned int, unsigned int, const std::vector<double> &, double )> makeCMLF( CalcMaxLengthFunc* cmlf )
            {
            return std::bind( &CalcMaxLengthFunc::operator(), cmlf, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3 );
            }
            %}''')

        # wrap NeuronGroup, SynapseGroup and CurrentSource
        pygennSmg.addSwigEnableUnderCaseConvert()
        
        pygennSmg.addSwigInclude( '"neuronGroup.h"' )
        pygennSmg.addSwigInclude( '"synapseGroup.h"' )
        pygennSmg.addSwigInclude( '"currentSource.h"' )

        with SwigAsIsScope( pygennSmg ):
            for mg in mgs:
                mg.models.append( 'Custom' )

        # wrap modelSpec.h
        pygennSmg.addSwigIgnore( 'init_connectivity()' )
        pygennSmg.addSwigIgnore( 'init_var()' )
        
        pygennSmg.addSwigInclude( '"modelSpec.h"' )
        pygennSmg.addSwigInclude( '"modelSpecInternal.h"' )

        # Loop through neuron models
        for n_model in mgs[0].models:
            # Ignore the overloads of the functions which automatically get an instance from class name
            ignore_base = 'ModelSpec::addNeuronPopulation<NeuronModels::{0}>(std::string const &,unsigned int,NeuronModels::{0}::ParamValues const &,NeuronModels::{0}::VarValues const &'.format(n_model)
            pygennSmg.addSwigIgnore(ignore_base + ")")
            pygennSmg.addSwigIgnore(ignore_base + ",int)")
            pygennSmg.addSwigIgnore(ignore_base + ",int,int)")

            # Add template expansion
            pygennSmg.addSwigTemplate(
                'ModelSpec::addNeuronPopulation<NeuronModels::{}>'.format(n_model ),
                'add_neuron_population_{}'.format( n_model ) )

        # Loop through all combinations of postsynaptic and weight update models
        for ps_model, wu_model in product(mgs[1].models, mgs[2].models):
            # Ignore the overloads of the functions which automatically get instances from class names
            ignore_base = "ModelSpec::addSynapsePopulation<WeightUpdateModels::{0},PostsynapticModels::{1}>(std::string const &,SynapseMatrixType,unsigned int,std::string const &,std::string const &,WeightUpdateModels::{0}::ParamValues const &,WeightUpdateModels::{0}::VarValues const &".format(
                wu_model, ps_model)
            ignore_wum_pre_post = "WeightUpdateModels::{0}::PreVarValues const &,WeightUpdateModels::{0}::PostVarValues const &".format(wu_model)
            ignore_psm = "PostsynapticModels::{0}::ParamValues const &,PostsynapticModels::{0}::VarValues const &".format(ps_model)
            ignore_connectivity = "InitSparseConnectivitySnippet::Init const &"
            pygennSmg.addSwigIgnore(ignore_base + "," + ignore_psm + ")")
            pygennSmg.addSwigIgnore(ignore_base + "," + ignore_psm + "," + ignore_connectivity + ")")
            pygennSmg.addSwigIgnore(ignore_base + "," + ignore_wum_pre_post + "," + ignore_psm + ")")
            pygennSmg.addSwigIgnore(ignore_base + "," + ignore_wum_pre_post + "," + ignore_psm + "," + ignore_connectivity + ")")

            # Add template expansion
            pygennSmg.addSwigTemplate(
                'ModelSpec::addSynapsePopulation<WeightUpdateModels::{0}, PostsynapticModels::{1}>'.format(
                    wu_model, ps_model),
                'add_synapse_population_{}_{}'.format(wu_model, ps_model))

        for cs_model in mgs[3].models:
            # Ignore the overload of the function which automatically gets instance from class name
            pygennSmg.addSwigIgnore("ModelSpec::addCurrentSource<CurrentSourceModels::{0}>(std::string const &,std::string const &,CurrentSourceModels::{0}::ParamValues const &,CurrentSourceModels::{0}::VarValues const &)".format(cs_model))
            pygennSmg.addSwigTemplate(
                'ModelSpec::addCurrentSource<CurrentSourceModels::{}>'.format(cs_model),
                'add_current_source_{}'.format(cs_model))

        pygennSmg.write( '\n// wrap variableMode.h.\n' )
        pygennSmg.addSwigIgnore( 'operator&' )
        pygennSmg.addSwigInclude( '"variableMode.h"' )

        pygennSmg.write( '\n// wrap synapseMatrixType.h\n' )
        pygennSmg.addSwigInclude( '"synapseMatrixType.h"' )


# if the module is called directly i.e. as $ python generate_swig_interfaces.py
if __name__ == '__main__':

    parser = ArgumentParser( description='Generate SWIG interfaces' )
    parser.add_argument('genn_path', type=str, help='Path to GeNN')

    gennPath = parser.parse_args().genn_path
    includePath = os.path.join( gennPath, 'include', 'genn' )
    
    # check that all required files can be found
    if not os.path.isfile( os.path.join( includePath, NEURONMODELS + '.h' ) ):
        print( 'Error: The {0} file is missing'.format( os.path.join( INDIR, NEURONMODELS + '.h' ) ) )
        exit(1)

    if not os.path.isfile( os.path.join( includePath, POSTSYNMODELS + '.h' ) ):
        print( 'Error: The {0} file is missing'.format( os.path.join( INDIR, POSTSYNMODELS + '.h' ) ) )
        exit(1)

    if not os.path.isfile( os.path.join(includePath, WUPDATEMODELS + '.h' ) ):
        print( 'Error: The {0} file is missing'.format( os.path.join( INDIR, WUPDATEMODELS + '.h' ) ) )
        exit(1)

    if not os.path.isfile( os.path.join(includePath, CURRSOURCEMODELS + '.h' ) ):
        print( 'Error: The {0} file is missing'.format( os.path.join( INDIR, CURRSOURCEMODELS + '.h' ) ) )
        exit(1)

    if not os.path.isfile( os.path.join(includePath, 'modelSpec.h' ) ):
        print( 'Error: The {0} file is missing'.format( os.path.join( INDIR, 'modelSpec.h' ) ) )
        exit(1)

    generateConfigs( gennPath )

