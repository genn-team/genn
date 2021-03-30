#pragma once

namespace CustomValues
{
//------------------------------------------------------------------------
// WUVarReferences
//------------------------------------------------------------------------
class WUVarReferences
{
public:
    WUVarReferences()
    {
    }
    
    WUVarReferences( const std::vector<Models::WUVarReference> &initialisers ) :
      m_Initialisers(initialisers.begin(), initialisers.end()){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    const std::vector<Models::WUVarReference> &getInitialisers() const
    {
        return m_Initialisers;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const Models::WUVarReference &operator[](size_t pos) const
    {
        return m_Initialisers[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::vector<Models::WUVarReference> m_Initialisers;
};
}
