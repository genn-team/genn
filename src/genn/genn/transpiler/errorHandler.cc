#include "transpiler/errorHandler.h"

// GeNN includes
#include "logging.h"

//----------------------------------------------------------------------------
// GeNN::Transpiler::ErrorHandler
//----------------------------------------------------------------------------
namespace GeNN::Transpiler
{
void ErrorHandler::error(size_t line, std::string_view message)
{
    report(line, "", message);
}
//----------------------------------------------------------------------------  
void ErrorHandler::error(const Token &token, std::string_view message)
{
    if(token.type == Token::Type::END_OF_FILE) {
        report(token.line, " at end", message);
    }
    else {
        report(token.line, " at '" + token.lexeme + "'", message);
    }
}
//----------------------------------------------------------------------------
void ErrorHandler::report(size_t line, std::string_view where, std::string_view message) 
{
    LOGE_TRANSPILER << "[line " << line << "] Error" << where << ": " << message;
    m_Error = true;
}

//----------------------------------------------------------------------------
// GeNN::Transpiler::SingleLineErrorHandler
//----------------------------------------------------------------------------
void SingleLineErrorHandler::error(size_t, std::string_view message)
{
    report("", message);
}
//----------------------------------------------------------------------------  
void SingleLineErrorHandler::error(const Token &token, std::string_view message)
{
    if(token.type == Token::Type::END_OF_FILE) {
        report(" at end", message);
    }
    else {
        report(" at '" + token.lexeme + "'", message);
    }
}
//----------------------------------------------------------------------------
void SingleLineErrorHandler::report(std::string_view where, std::string_view message) 
{
    LOGE_TRANSPILER << "Error" << where << ": " << message;
    m_Error = true;
}
}   // namespace GeNN::Transpiler
