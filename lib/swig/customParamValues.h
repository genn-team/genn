#ifndef CUSTOMPARAMVALUES_H
#define CUSTOMPARAMVALUES_H
namespace CustomValues
{
class ParamValues
{
public:
    template<typename T>
    ParamValues( const std::vector<T> &vals) : m_Values( vals )
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
}
#endif // CUSTOMPARAMVALUES_H
