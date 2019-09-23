#pragma once

// GeNN userproject includes
#include "../../../userproject/include/sharedLibraryModel.h"

//----------------------------------------------------------------------------
// SharedLibraryModelNumpy
//----------------------------------------------------------------------------
template<typename scalar = float>
class SharedLibraryModelNumpy : public SharedLibraryModel<scalar>
{
public:
    SharedLibraryModelNumpy()
    {
    }

    SharedLibraryModelNumpy(const std::string &pathToModel, const std::string &modelName)
    :   SharedLibraryModel<scalar>(pathToModel, modelName)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    // Assign symbol from shared model to the provided pointer.
    // The symbol is supposed to be an array
    // When used with numpy, wrapper automatically provides varPtr and n1
    template<typename T>
    void assignExternalPointerArray(const std::string &varName, const int varSize, T** varPtr, int* n1)
    {
        *varPtr = this->template getArray<T>(varName);
        *n1 = varSize;
    }
    
    // Assign symbol from shared model to the provided pointer.
    // The symbol is supposed to be a single value
    // When used with numpy, wrapper automatically provides varPtr and n1
    template<typename T>
    void assignExternalPointerSingle(const std::string &varName, T** varPtr, int* n1)
    {
        *varPtr = this->template getScalar<T>(varName);
        *n1 = 1;
    }

private:
    // Hide C++ based public API
    using SharedLibraryModel<scalar>::getSymbol;
    using SharedLibraryModel<scalar>::getArray;
    using SharedLibraryModel<scalar>::getScalar;
};
