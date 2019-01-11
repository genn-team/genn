#pragma once

namespace CustomValues
{
//------------------------------------------------------------------------
// VarValues
//------------------------------------------------------------------------
class VarValues
{
private:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<Models::VarInit> InitialiserArray;

public:
    VarValues()
    {
    }
    
    template<typename T>
    VarValues( const std::vector<T> &initialisers ) :
      m_Initialisers(initialisers.begin(), initialisers.end()){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    const std::vector<Models::VarInit> &getInitialisers() const
    {
        return m_Initialisers;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const Models::VarInit &operator[](size_t pos) const
    {
        return m_Initialisers[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    InitialiserArray m_Initialisers;
};
}
