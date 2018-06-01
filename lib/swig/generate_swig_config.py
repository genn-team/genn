import sys
from os import path

FNEURONMODELS = 'include/newNeuronModels.h'
FPOSTSYNMODELS = 'include/newPostsynapticModels.h'
FWUPDATEMODELS = 'include/newWeightUpdateModels.h'
FSWIG_LIBGENN = 'swig/libgenn'
FSWIG_NEURONMODELS = 'swig/newNeuronModels'
FSWIG_POSTSYNMODELS = 'swig/newPostsynapticModels'
FSWIG_WUPDATEMODELS = 'swig/newWeightUpdateModels'

def generate_makers( nspace, model_name, num_params, num_vars ):
    return generate_value_maker( nspace, model_name, 'ParamValues', num_params ) + \
            generate_value_maker( nspace, model_name, 'VarValues', num_vars )


def generate_value_maker( nspace, model_name, value_name, num_values ):

    nspace_model_name = nspace + '::' + model_name
    model_value_maker = '{0}::{2}* make_{1}_{2}('.format( nspace_model_name, model_name, value_name )
    for i in range( num_values - 1 ):
        model_value_maker += 'double v' + str( i ) + ', '
    if num_values > 0:
        model_value_maker += 'double v' + str( num_values - 1 )
    model_value_maker += ') {\n'
    model_value_maker += 'return new {0}::{1}('.format( nspace_model_name, value_name )
                    
    for i in range( num_values - 1 ):
        model_value_maker += 'v' + str( i ) + ', '
    if num_values > 0:
        model_value_maker += 'v' + str( num_values - 1 )

    model_value_maker += ');\n}\n'

    return model_value_maker

def generate_configs( genn_lib_path ):
    with open( path.join( genn_lib_path, FNEURONMODELS ), 'r' ) as neuronModels_h, \
            open( path.join( genn_lib_path, FPOSTSYNMODELS ), 'r' ) as postsynModels_h, \
            open( path.join( genn_lib_path, FWUPDATEMODELS ), 'r' ) as wUpdateModels_h:
        with open( path.join( genn_lib_path, FSWIG_LIBGENN + '.src' ), 'r' ) as libgenn_src, \
                open( path.join( genn_lib_path, FSWIG_NEURONMODELS + '.src' ), 'r' ) as neuronModels_src, \
                open( path.join( genn_lib_path, FSWIG_POSTSYNMODELS + '.src' ), 'r' ) as postsynModels_src, \
                open( path.join( genn_lib_path, FSWIG_WUPDATEMODELS + '.src' ), 'r' ) as wUpdateModels_src:
            with open( path.join( genn_lib_path, 'libgenn.i' ), 'w' ) as libgenn_i, \
                    open( path.join( genn_lib_path, FSWIG_NEURONMODELS + '.i' ), 'w' ) as neuronModels_i, \
                    open( path.join( genn_lib_path, FSWIG_POSTSYNMODELS + '.i' ), 'w' ) as postsynModels_i, \
                    open( path.join( genn_lib_path, FSWIG_WUPDATEMODELS + '.i' ), 'w' ) as wUpdateModels_i:
            
                libgenn_i.writelines( libgenn_src.readlines() )
                neuronModels_i.writelines( neuronModels_src.readlines() )
                postsynModels_i.writelines( postsynModels_src.readlines() )
                wUpdateModels_i.writelines( wUpdateModels_src.readlines() )
            
                neuronModels_i.write( '%inline %{\n' )
                postsynModels_i.write( '%inline %{\n' )
                wUpdateModels_i.write( '%inline %{\n' )

                supported_neurons = []
            
                for line in neuronModels_h.readlines():
                    line = line.lstrip()
                    if line.startswith( 'DECLARE_MODEL(' ):
                        nspace_model_name, num_params, num_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )

                        num_params = int( num_params)
                        num_vars = int( num_vars )
                        model_name = nspace_model_name.split( '::' )[1]

                        supported_neurons.append( model_name )

                        nnModel_template = '%template(addNeuronPopulation_{0}) NNmodel::addNeuronPopulation<{1}>;\n'.format( model_name, nspace_model_name )

                        libgenn_i.write( nnModel_template )
                        neuronModels_i.write( generate_makers( 'NeuronModels', model_name, num_params, num_vars ) )
                
                neuronModels_i.write( 'std::vector< std::string > getSupportedNeurons() {\nreturn std::vector<std::string>{"' + '", "'.join(supported_neurons) + '"};\n}\n' )

                neuronModels_i.write( '%}\n' )

                postsyn_models = []
            
                for line in postsynModels_h.readlines():
                    line = line.lstrip()
                    if line.startswith( 'DECLARE_MODEL(' ):
                        model_name, num_params, num_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )

                        num_params = int( num_params)
                        num_vars = int( num_vars )

                        postsyn_models.append( model_name )

                        postsynModels_i.write( generate_makers( 'PostsynapticModels', model_name, num_params, num_vars ) )

                postsynModels_i.write( 'std::vector< std::string > getSupportedPostsyn() {\nreturn std::vector<std::string>{"' + '", "'.join(postsyn_models) + '"};\n}\n' )
                postsynModels_i.write( '%}\n' )

                wupdate_models = []
            
                for line in wUpdateModels_h.readlines():
                    line = line.lstrip()
                    if line.startswith( 'DECLARE_MODEL(' ):
                        model_name, num_params, num_vars = line.split( '(' )[1].split( ')' )[0].split( ',' )

                        num_params = int( num_params)
                        num_vars = int( num_vars )

                        wupdate_models.append( model_name )

                        wUpdateModels_i.write( generate_makers( 'WeightUpdateModels', model_name, num_params, num_vars ) )

                wUpdateModels_i.write( 'std::vector< std::string > getSupportedWUpdate() {\nreturn std::vector<std::string>{"' + '", "'.join(wupdate_models) + '"};\n}\n' )
                wUpdateModels_i.write( '%}\n' )


                for ps_model in postsyn_models:
                    for wu_model in wupdate_models:

                        nnModel_template = '%template(addSynapsePopulation_{0}_{1}) NNmodel::addSynapsePopulation<WeightUpdateModels::{0}, PostsynapticModels::{1}>;\n'.format( wu_model, ps_model )

                        libgenn_i.write( nnModel_template )


if __name__ == '__main__':
    try:
        genn_lib_path = sys.argv[1]
    except:
        print( 'Error: A path to GeNN lib required' )
        print( 'Example usage: python genn_lib_path GENN_PATH/lib' )
        exit(1)

    if not path.isfile( path.join( genn_lib_path, FNEURONMODELS ) ):
        print( 'Error: the {0} file is missing', FNEURONMODELS )
        exit(1)

    if not path.isfile( path.join( genn_lib_path, FPOSTSYNMODELS ) ):
        print( 'Error: the {0} file is missing', FPOSTSYNMODELS )
        exit(1)
    
    if not path.isfile( path.join(genn_lib_path, FWUPDATEMODELS ) ):
        print( 'Error: the {0} file is missing', FWUPDATEMODELS )
        exit(1)

    if not path.isfile( path.join( genn_lib_path, FSWIG_LIBGENN + '.src' ) ):
        print( 'Error: the {0} file is missing', FSWIG_LIBGENN + '.src' )
        exit(1)

    if not path.isfile( path.join( genn_lib_path, FSWIG_NEURONMODELS + '.src' ) ):
        print( 'Error: the {0} file is missing', FSWIG_NEURONMODELS + '.src' )
        exit(1)

    if not path.isfile( path.join( genn_lib_path, FSWIG_POSTSYNMODELS + '.src' ) ):
        print( 'Error: the {0} file is missing', FSWIG_POSTSYNMODELS + '.src' )
        exit(1)
    
    if not path.isfile( path.join( genn_lib_path, FSWIG_WUPDATEMODELS + '.src' ) ):
        print( 'Error: the {0} file is missing', FSWIG_WUPDATEMODELS + '.src' )
        exit(1)

    generate_configs( genn_lib_path )

