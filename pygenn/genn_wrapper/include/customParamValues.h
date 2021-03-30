#pragma once

namespace CustomValues
{
class ParamValues
{
public:
    ParamValues()
    {
    }
    
    ParamValues( const std::vector<double> &vals) : m_Values( vals )
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    const std::vector<double> &getInitialisers() const
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
