%module SharedLibraryModel

%{
#define SWIG_FILE_WITH_INIT
#include "SharedLibraryModel.h"
%}


%include <std_string.i>
%include <std_pair.i>
%include <std_vector.i>
/* %include <std_iostream.i> */
%include "numpy.i"
%init %{
import_array();
%}

/* %template(DoubleVector) std::vector<double>; */
/* %template(DoubleVector2D) std::vector<std::vector<double>>; */
/* %template(FloatVector) std::vector<float>; */
/* %template(FloatVector2D) std::vector<std::vector<float>>; */
%template() std::pair<std::string, std::string>;
%template(StringPairVector) std::vector<std::pair<std::string, std::string>>;

/* wrapping of numpy arrays */
%apply ( double* IN_ARRAY1, int DIM1 ) {( double* initData, int n1 )};
%apply ( float* IN_ARRAY1, int DIM1 ) {( float* initData, int n1 )};
%apply ( unsigned int* IN_ARRAY1, int DIM1) {( unsigned int* initData, int n1 )};
%apply ( double* INPLACE_ARRAY2, int DIM1, int DIM2 ) {( double* out, int n1, int n2 )};
%apply ( float* INPLACE_ARRAY2, int DIM1, int DIM2 ) {( float* out, int n1, int n2 )};

%include "SharedLibraryModel.h"
%template(SharedLibraryModel_f) SharedLibraryModel<float>;
%template(SharedLibraryModel_d) SharedLibraryModel<double>;

