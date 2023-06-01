#pragma once

namespace CustomValues
{
//------------------------------------------------------------------------
// EGPReferences
//------------------------------------------------------------------------
class EGPReferences
{
public:
    EGPReferences()
    {
    }
    
    EGPReferences( const std::vector<Models::EGPReference> &initialisers ) :
      m_Initialisers(initialisers.begin(), initialisers.end()){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    const std::vector<Models::EGPReference> &getInitialisers() const
    {
        return m_Initialisers;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const Models::EGPReference &operator[](size_t pos) const
    {
        return m_Initialisers[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::vector<Models::EGPReference> m_Initialisers;
};
}
