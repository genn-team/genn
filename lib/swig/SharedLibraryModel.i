%module SharedLibraryModel

%{
#define SWIG_FILE_WITH_INIT // for numpy
#include "SharedLibraryModel.h"
%}

%include <std_string.i>

%include "numpy.i"
%init %{
import_array();
%}

/* wrapping of numpy arrays */
%apply ( double** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( double** varPtr, int* n1 )};
%apply ( float** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( float** varPtr, int* n1 )};
%apply ( double** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( double** tPtr, int* n1 )};
%apply ( float** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( float** tPtr, int* n1 )};
%apply ( unsigned int** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned int** spkPtr, int* n1 )};
%apply ( unsigned long long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned long long** timestepPtr, int* n1 )};

%include "SharedLibraryModel.h"
%template(SharedLibraryModel_f) SharedLibraryModel<float>;
%template(SharedLibraryModel_d) SharedLibraryModel<double>;

