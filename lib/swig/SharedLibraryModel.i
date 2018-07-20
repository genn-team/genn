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
%apply ( unsigned long long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {( unsigned long long** varPtr, int* n1 )};

%apply ( unsigned int* IN_ARRAY1, int DIM1 ) {( unsigned int* _ind, int nConn )};
%apply ( unsigned int* IN_ARRAY1, int DIM1 ) {( unsigned int* _indInG, int nPre )};
%apply ( double* IN_ARRAY1, int DIM1 ) {( double* _g, int nG )};
%apply ( float* IN_ARRAY1, int DIM1 ) {( float* _g, int nG )};

%include "SharedLibraryModel.h"
%template(assignExternalPointerArray_f) SharedLibraryModel::assignExternalPointerArray<float>;
%template(assignExternalPointerArray_d) SharedLibraryModel::assignExternalPointerArray<double>;
%template(assignExternalPointerArray_ld) SharedLibraryModel::assignExternalPointerArray<long double>;
%template(assignExternalPointerArray_i) SharedLibraryModel::assignExternalPointerArray<int>;
%template(assignExternalPointerArray_ui) SharedLibraryModel::assignExternalPointerArray<unsigned int>;
%template(assignExternalPointerSingle_f) SharedLibraryModel::assignExternalPointerSingle<float>;
%template(assignExternalPointerSingle_d) SharedLibraryModel::assignExternalPointerSingle<double>;
%template(assignExternalPointerSingle_ld) SharedLibraryModel::assignExternalPointerSingle<long double>;
%template(assignExternalPointerSingle_i) SharedLibraryModel::assignExternalPointerSingle<int>;
%template(assignExternalPointerSingle_ui) SharedLibraryModel::assignExternalPointerSingle<unsigned int>;
%template(assignExternalPointerSingle_ull) SharedLibraryModel::assignExternalPointerSingle<unsigned long long>;

%template(SharedLibraryModel_f) SharedLibraryModel<float>;
%template(SharedLibraryModel_d) SharedLibraryModel<double>;
%template(SharedLibraryModel_ld) SharedLibraryModel<long double>;

