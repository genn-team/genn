#ifndef CUSTOMVARVALUES_H
#define CUSTOMVARVALUES_H
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
    typedef std::vector<NewModels::VarInit> InitialiserArray;

public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<VarInit, 4> can be initialized with <= 4 elements
    template<typename T>
    VarValues( const std::vector<T> &initialisers ) :
      m_Initialisers(initialisers.begin(), initialisers.end()){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    const std::vector<NewModels::VarInit> &getInitialisers() const
    {
        return m_Initialisers;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const NewModels::VarInit &operator[](size_t pos) const
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
#endif // CUSTOMVARVALUES_H
