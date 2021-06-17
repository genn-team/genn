#pragma once

namespace CustomValues
{
//------------------------------------------------------------------------
// VarReferences
//------------------------------------------------------------------------
class VarReferences
{
public:
    VarReferences()
    {
    }
    
    VarReferences( const std::vector<Models::VarReference> &initialisers ) :
      m_Initialisers(initialisers.begin(), initialisers.end()){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    const std::vector<Models::VarReference> &getInitialisers() const
    {
        return m_Initialisers;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const Models::VarReference &operator[](size_t pos) const
    {
        return m_Initialisers[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::vector<Models::VarReference> m_Initialisers;
};
}
