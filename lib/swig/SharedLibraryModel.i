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
%apply ( long double** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( long double** varPtr, int* n1 )};
%apply ( unsigned int** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned int** varPtr, int* n1 )};
%apply ( int** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( int** varPtr, int* n1 )};

%apply ( double** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( double** tPtr, int* n1 )};
%apply ( float** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( float** tPtr, int* n1 )};
%apply ( unsigned int** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned int** spkPtr, int* n1 )};
%apply ( unsigned long long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned long long** timestepPtr, int* n1 )};
%apply ( unsigned int* IN_ARRAY1, int DIM1 ) {( unsigned int* _ind, int nConn )};
%apply ( unsigned int* IN_ARRAY1, int DIM1 ) {( unsigned int* _indInG, int nPre )};
%apply ( double* IN_ARRAY1, int DIM1 ) {( double* _g, int nG )};
%apply ( float* IN_ARRAY1, int DIM1 ) {( float* _g, int nG )};

%include "SharedLibraryModel.h"
%template(assignExternalPointer_f) SharedLibraryModel::assignExternalPointer<float>;
%template(assignExternalPointer_d) SharedLibraryModel::assignExternalPointer<double>;
%template(assignExternalPointer_ld) SharedLibraryModel::assignExternalPointer<long double>;
%template(assignExternalPointer_i) SharedLibraryModel::assignExternalPointer<int>;
%template(assignExternalPointer_ui) SharedLibraryModel::assignExternalPointer<unsigned int>;

%template(SharedLibraryModel_f) SharedLibraryModel<float>;
%template(SharedLibraryModel_d) SharedLibraryModel<double>;
%template(SharedLibraryModel_ld) SharedLibraryModel<long double>;

