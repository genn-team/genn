
namespace CustomValues
{
class ParamValues
{
public:
    template<typename T>
    ParamValues( std::vector<T> &vals) : m_Values( vals )
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    const std::vector<double> &getValues() const
    {
        return m_Values;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    double operator[](size_t pos) const
    {
        return m_Values[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::vector<double> m_Values;
};
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
    VarValues( std::vector<T> &initialisers ) : m_Initialisers(InitialiserArray( initialisers.begin(), initialisers.end() )) {}

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

