#pragma once

// Standard C++ includes
#include <string_view>

// Transpiler includes
#include "transpiler/token.h"

//---------------------------------------------------------------------------
// GeNN::Transpiler::ErrorHandlerBase
//---------------------------------------------------------------------------
namespace GeNN::Transpiler
{
class ErrorHandlerBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void error(size_t line, std::string_view message) = 0;
    virtual void error(const Token &token, std::string_view message) = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::ErrorHandler
//---------------------------------------------------------------------------
class ErrorHandler : public ErrorHandlerBase
{
public:
    ErrorHandler() : m_Error(false)
    {
    }

    //------------------------------------------------------------------------
    // ErrorHandlerBase virtuals
    //------------------------------------------------------------------------
    virtual void error(size_t line, std::string_view message) final;
    virtual void error(const Token &token, std::string_view message) final;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool hasError() const { return m_Error; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void report(size_t line, std::string_view where, std::string_view message);
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    bool m_Error;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::SingleLineErrorHandler
//---------------------------------------------------------------------------
class SingleLineErrorHandler : public ErrorHandlerBase
{
public:
    SingleLineErrorHandler() : m_Error(false)
    {
    }

    //------------------------------------------------------------------------
    // ErrorHandlerBase virtuals
    //------------------------------------------------------------------------
    virtual void error(size_t line, std::string_view message) final;
    virtual void error(const Token &token, std::string_view message) final;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool hasError() const { return m_Error; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void report(std::string_view where, std::string_view message);
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    bool m_Error;
};
}   // namespace GeNN::Transpiler
