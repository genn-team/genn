import sys
import os
from string import Template

NEURONMODELS = 'newNeuronModels'
POSTSYNMODELS = 'newPostsynapticModels'
WUPDATEMODELS = 'newWeightUpdateModels'
CURRSOURCEMODELS = 'currentSourceModels'
NNMODEL = 'modelSpec'
MAIN_MODULE = 'pygenn'
INDIR = 'include/'
SWIGDIR = 'swig/'

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
class CppIncludeGuardScope( object ):
    def __init__( self, ofs, guard ):
        '''Adds an include guard block'''
        self.os = ofs
        self.guard = guard
    def __enter__( self ):
        self.os.write( '#ifndef {0}\n#define {0}\n\n'.format( self.guard ) )
    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.write( '#endif // {}\n'.format( self.guard ) )

class SwigModuleGenerator( object ):

    def __init__( self, moduleName, oFile ):
        '''A helper class for generating SWIG interface files'''
        self.name = moduleName
        self.oFile = oFile

    def __enter__(self):
        self.os = open( self.oFile, 'w' )
        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self.os.close()

    def addSwigModuleHeadline( self, directors=False, comment='' ):
        '''Adds a line naming a module and enabling directors feature for inheritance in python (disabled by default)'''
        directorsCode = ''
        if directors:
            directorsCode = '(directors="1")'
        self.write( '%module{} {} {}\n'.format( directorsCode, self.name, comment ) )

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

    vals = 'vals'
    if numValues == 0:
        vals = ''

    mg.write( 'static {0}::{1}::{2}* make_{2}( const std::vector<double> & {3} )'.format(
        mg.name,
        modelName,
        valueName,
        vals
        ) )
    with CppBlockScope( mg ):
        if numValues != 0:
            mg.write( 'double ' + ', '.join( ['v{0} = vals[{0}]'.format( i )
                                          for i in range(numValues)] ) + ';\n' )
        mg.write( 'return new {}::{}({});\n'.format(
            mg.name + '::' + modelName,
            valueName,
            ', '.join( [('v' + str( i )) for i in range(numValues)] ) ) )


def generateCustomClassDeclaration( nSpace ):

    return Template('''
namespace ${NAMESPACE}
{
class Custom : public Base
{
private:
    static ${NAMESPACE}::Custom *s_Instance;
public:
    static const ${NAMESPACE}::Custom *getInstance()
    {
        if ( s_Instance == NULL )
        {
            s_Instance = new ${NAMESPACE}::Custom;
        }
        return s_Instance;
    }
    typedef CustomValues::ParamValues ParamValues;
    typedef CustomValues::VarValues VarValues;
    static CustomValues::ParamValues* make_ParamValues( const std::vector< double > & vals )
    {
        return new CustomValues::ParamValues( vals );
    }
    static CustomValues::VarValues* make_VarValues( const std::vector< double > & vals )
    {
        return new CustomValues::VarValues( vals );
    }
};
} // namespace ${NAMESPACE}
''').substitute(NAMESPACE=nSpace)

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


def generateSharedLibraryModelInterface( genn_lib_path ):
    '''Generates SharedLibraryModel.i file'''
    with SwigModuleGenerator('SharedLibraryModel',
            os.path.join( genn_lib_path, SWIGDIR, 'SharedLibraryModel.i' ) ) as mg:
        mg.addAutoGenWarning()
        mg.addSwigModuleHeadline()
        with SwigAsIsScope( mg ):
            mg.write( '#define SWIG_FILE_WITH_INIT // for numpy\n' )
            mg.addCppInclude( '"SharedLibraryModel.h"' )

        mg.addSwigInclude( '<std_string.i>' )
        mg.addSwigInclude( '"numpy.i"' )
        mg.write( '%numpy_typemaps(long double, NPY_LONGDOUBLE, int) ')
        with SwigInitScope( mg ):
            mg.write( 'import_array();\n')

        # These are all data types supported by numpy SWIG interface (at least by default) plus long double
        npDTypes = (
                'signed char', 'unsigned char',
                'short', 'unsigned short',
                'int', 'unsigned int',
                'long', 'unsigned long',
                'long long', 'unsigned long long',
                'float', 'double', 'long double'
        )

        for dataType in [dt+'*' for dt in npDTypes]:
            mg.write( generateNumpyApplyArgoutviewArray1D( dataType, 'varPtr', 'n1' ) )
        mg.write( generateNumpyApplyInArray1D( 'unsigned int*', '_ind', 'nConn' ) )
        mg.write( generateNumpyApplyInArray1D( 'unsigned int*', '_indInG', 'nPre' ) )
        mg.write( generateNumpyApplyInArray1D( 'double*', '_g', 'nG' ) )
        mg.write( generateNumpyApplyInArray1D( 'float*', '_g', 'nG' ) )

        mg.addSwigInclude( '"SharedLibraryModel.h"' )
        for dtShort, dataType in zip( [ "".join([dt_[0] for dt_ in dt.split()]) for dt in npDTypes],
                npDTypes ):
            mg.addSwigTemplate( 'SharedLibraryModel::assignExternalPointerArray<{}>'.format( dataType ),
                'assignExternalPointerArray_' + dtShort )
            mg.addSwigTemplate( 'SharedLibraryModel::assignExternalPointerSingle<{}>'.format( dataType ),
                'assignExternalPointerSingle_' + dtShort )

        for dtShort, dataType in zip( ('f', 'd', 'ld'), ('float', 'double', 'long double') ):
            mg.addSwigTemplate( 'SharedLibraryModel<{}>'.format( dataType ),
                'SharedLibraryModel_' + dtShort )


def generateStlContainersInterface( genn_lib_path ):
    '''Generates headers/sources with Custom classes'''
    # generate Custom models
    with SwigModuleGenerator( 'StlContainers',
            os.path.join( genn_lib_path, SWIGDIR, 'stl_containers.i' ) ) as mg:
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
        mg.addSwigTemplate( 'std::pair<std::string, std::string>', 'StringPair' )
        mg.addSwigTemplate( 'std::pair<std::string, double>', 'StringDoublePair' )
        mg.addSwigTemplate( 'std::pair<std::string, std::pair<std::string, double>>',
                'StringStringDoublePairPair' )
        mg.addSwigTemplate( 'std::pair<std::string, std::{}>'.format( dpfunc_template_spec ),
                'StringDPFPair')
        mg.addSwigTemplate( 'std::vector<std::string>', 'StringVector' )
        mg.addSwigTemplate( 'std::vector<std::pair<std::string, std::string>>', 'StringPairVector' )
        mg.addSwigTemplate( 'std::vector<std::pair<std::string, std::pair<std::string, double>>>',
                'StringStringDoublePairPairVector' )
        mg.addSwigTemplate( 'std::vector<std::pair<std::string, std::{}>>'.format( dpfunc_template_spec ),
                'StringDPFPairVector' )

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


def generateCustomModelDeclImpls( genn_lib_path ):
    '''Generates headers/sources with Custom classes'''
    # generate Custom models
    models = [NEURONMODELS, POSTSYNMODELS, WUPDATEMODELS, CURRSOURCEMODELS]
    for model in models:
        nSpace = model[:]
        if model.startswith('new'):
            nSpace = nSpace[3:]
        else:
            nSpace = 'C' + nSpace[1:]
        with SwigModuleGenerator( 'decl',
                os.path.join( genn_lib_path, SWIGDIR, model + 'Custom.h' ) ) as mg:
            mg.addAutoGenWarning()
            with CppIncludeGuardScope( mg, model.upper() + 'CUSTOM_H'):
                mg.addCppInclude( '"' + model + '.h"' )
                mg.addCppInclude( '"customValues.h"' )
                mg.write(generateCustomClassDeclaration(nSpace))
        with SwigModuleGenerator( 'impl',
                os.path.join( genn_lib_path, SWIGDIR, model + 'Custom.cc' ) ) as mg:
            mg.addAutoGenWarning()
            mg.addCppInclude( '"' + model + 'Custom.h"' )
            mg.write('IMPLEMENT_MODEL({}::Custom);\n'.format(nSpace))


def generateConfigs( genn_lib_path ):
    '''Generates SWIG interfaces'''
    generateStlContainersInterface( genn_lib_path )
    generateCustomModelDeclImpls( genn_lib_path )
    generateSharedLibraryModelInterface( genn_lib_path )

    # open header files with models and instantiate SwigModuleGenerators
    with open( os.path.join( genn_lib_path, INDIR + NEURONMODELS + ".h" ), 'r' ) as neuronModels_h, \
            open( os.path.join( genn_lib_path, INDIR + POSTSYNMODELS + ".h" ), 'r' ) as postsynModels_h, \
            open( os.path.join( genn_lib_path, INDIR + WUPDATEMODELS + ".h" ), 'r' ) as wUpdateModels_h, \
            open( os.path.join( genn_lib_path, INDIR + CURRSOURCEMODELS + ".h" ), 'r' ) as currSrcModels_h, \
            SwigModuleGenerator( MAIN_MODULE,
                    os.path.join( genn_lib_path, MAIN_MODULE + '.i' ) ) as pygennSmg, \
            SwigModuleGenerator( 'NeuronModels',
                    os.path.join( genn_lib_path, SWIGDIR + NEURONMODELS + ".i" ) ) as neuronSmg, \
            SwigModuleGenerator( 'PostsynapticModels',
                    os.path.join( genn_lib_path, SWIGDIR + POSTSYNMODELS + ".i" ) ) as postsynSmg, \
            SwigModuleGenerator( 'WeightUpdateModels',
                    os.path.join( genn_lib_path, SWIGDIR + WUPDATEMODELS + ".i" ) ) as wUpdateSmg, \
            SwigModuleGenerator( 'CurrentSourceModels',
                    os.path.join( genn_lib_path, SWIGDIR + CURRSOURCEMODELS + ".i" ) ) as currSrcSmg:

        mgs = [ neuronSmg, postsynSmg, wUpdateSmg, currSrcSmg ]

        pygennSmg.addAutoGenWarning()
        pygennSmg.addSwigModuleHeadline()
        with SwigAsIsScope( pygennSmg ):
            pygennSmg.addCppInclude( '"variableMode.h"' )
            pygennSmg.addCppInclude( '"global.h"' )
            pygennSmg.addCppInclude( '"modelSpec.h"' )
            pygennSmg.addCppInclude( '"generateALL.h"' )
            pygennSmg.addCppInclude( '"synapseMatrixType.h"' )
            pygennSmg.addCppInclude( '"neuronGroup.h"' )
            pygennSmg.addCppInclude( '"synapseGroup.h"' )
            pygennSmg.addCppInclude( '"currentSource.h"' )

            pygennSmg.addCppInclude( '"neuronGroup.h"' )
            pygennSmg.addCppInclude( '"synapseGroup.h"' )
            pygennSmg.addCppInclude( '"currentSource.h"' )
            pygennSmg.addCppInclude( '"' + NNMODEL + '.h"' )
            for header in (NEURONMODELS, POSTSYNMODELS,
                           WUPDATEMODELS, CURRSOURCEMODELS):
                pygennSmg.addCppInclude( '"' + header + 'Custom.h"' )

        pygennSmg.addSwigImport( '"swig/stl_containers.i"' )

        for mg, header in zip(mgs, (neuronModels_h, postsynModels_h,
                                   wUpdateModels_h, currSrcModels_h)):
            _, headerFilename = os.path.split( header.name )
            pygennSmg.addSwigImport( '"swig/' + headerFilename.split('.')[0] + '.i"' )
            mg.addAutoGenWarning()
            mg.addSwigModuleHeadline( directors = True )
            with SwigAsIsScope( mg ):
                mg.addCppInclude( '"' + headerFilename + '"' )
                mg.addCppInclude( '"' + headerFilename.split('.')[0] + 'Custom.h"' )
                mg.addCppInclude( '"../swig/customValues.h"' )
            mg.addSwigIgnore( 'LegacyWrapper' )
            mg.addSwigImport( '"newModels.i"' )
            mg.addSwigFeatureDirector( mg.name + '::Base' )
            mg.addSwigInclude( '"include/' + headerFilename + '"' )
            mg.addSwigFeatureDirector( mg.name + '::Custom' )
            mg.addSwigInclude( '"' + headerFilename.split('.')[0] + 'Custom.h"' )

            mg.models = []
            for line in header.readlines():
                line = line.lstrip()
                if line.startswith( 'DECLARE_MODEL(' ):
                    nspace_model_name, num_params, num_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )
                    if mg.name == 'NeuronModels':
                        model_name = nspace_model_name.split( '::' )[1]
                    else:
                        model_name = nspace_model_name

                    num_params = int( num_params)
                    num_vars = int( num_vars )

                    mg.models.append( model_name )

                    with SwigExtendScope( mg, mg.name + '::' + model_name ):
                        writeValueMakerFunc( model_name, 'ParamValues', num_params, mg )
                        writeValueMakerFunc( model_name, 'VarValues', num_vars, mg )

        pygennSmg.addSwigInclude( '"include/neuronGroup.h"' )
        pygennSmg.addSwigInclude( '"include/synapseGroup.h"' )
        pygennSmg.addSwigInclude( '"include/currentSource.h"' )

        with SwigAsIsScope( pygennSmg ):
            for mg in mgs:
                mg.models.append( 'Custom' )

        pygennSmg.addSwigIgnore( 'initGeNN' )
        pygennSmg.addSwigIgnore( 'GeNNReady' )
        pygennSmg.addSwigIgnore( 'SynapseConnType' )
        pygennSmg.addSwigIgnore( 'SynapseGType' )
        pygennSmg.addSwigInclude( '"include/' + NNMODEL + '.h"' )

        for n_model in mgs[0].models:
            pygennSmg.addSwigTemplate(
                'NNmodel::addNeuronPopulation<{}::{}>'.format( mgs[0].name, n_model ),
                'addNeuronPopulation_{}'.format( n_model ) )

        for ps_model in mgs[1].models:
            for wu_model in mgs[2].models:

                pygennSmg.addSwigTemplate(
                    'NNmodel::addSynapsePopulation<{}::{}, {}::{}>'.format(
                        mgs[2].name, wu_model, mgs[1].name, ps_model ),
                    'addSynapsePopulation_{}_{}'.format( wu_model, ps_model ) )

        for cs_model in mgs[3].models:
            pygennSmg.addSwigTemplate(
                'NNmodel::addCurrentSource<{}::{}>'.format( mgs[3].name, cs_model ),
                'addCurrentSource_{}'.format( cs_model ) )

        pygennSmg.write( '\n// wrap necessary functions from generateALL.h\n' )
        pygennSmg.addSwigIgnore( 'generate_model_runner' )
        pygennSmg.addSwigInclude( '"include/generateALL.h"' )

        pygennSmg.write( '\n// wrap variables from global.h. Note that GENN_PREFERENCES is' )
        pygennSmg.write( '// already covered in the genn_preferences.i interface\n' )
        pygennSmg.addSwigIgnore( 'GENN_PREFERENCES' )
        pygennSmg.addSwigIgnore( 'deviceProp' )
        pygennSmg.addSwigInclude( '"include/global.h"' )
        pygennSmg.addSwigIgnore( 'operator&' )
        pygennSmg.addSwigInclude( '"include/variableMode.h"' )

        pygennSmg.write( '\n// wrap synapseMatrixType.h\n' )
        pygennSmg.addSwigInclude( '"include/synapseMatrixType.h"' )
        with SwigInlineScope( pygennSmg ):
            pygennSmg.write( 'void setDefaultVarMode( const VarMode &varMode ) {\n' )
            pygennSmg.write( '  GENN_PREFERENCES::defaultVarMode = varMode;\n}' )
        pygennSmg.addSwigImport( '"swig/genn_preferences.i"' )


if __name__ == '__main__':
    try:
        genn_lib_path = sys.argv[1]
    except:
        print( 'Error: A path to GeNN lib required' )
        print( 'Example usage: python genn_lib_path GENN_PATH/lib' )
        exit(1)

    if not os.path.isfile( os.path.join( genn_lib_path, INDIR + NEURONMODELS + '.h' ) ):
        print( 'Error: the {0} file is missing'.format( INDIR + NEURONMODELS + '.h' ) )
        exit(1)

    if not os.path.isfile( os.path.join( genn_lib_path, INDIR + POSTSYNMODELS + '.h' ) ):
        print( 'Error: the {0} file is missing'.format( INDIR + POSTSYNMODELS + '.h' ) )
        exit(1)
    
    if not os.path.isfile( os.path.join(genn_lib_path, INDIR + WUPDATEMODELS + '.h' ) ):
        print( 'Error: the {0} file is missing'.format( INDIR + WUPDATEMODELS + '.h' ) )
        exit(1)

    if not os.path.isfile( os.path.join(genn_lib_path, INDIR + CURRSOURCEMODELS + '.h' ) ):
        print( 'Error: the {0} file is missing'.format( INDIR + CURRSOURCEMODELS + '.h' ) )
        exit(1)

    if not os.path.isfile( os.path.join(genn_lib_path, INDIR + NNMODEL + '.h' ) ):
        print( 'Error: the {0} file is missing'.format( INDIR + NNMODEL + '.h' ) )
        exit(1)

    generateConfigs( genn_lib_path )

