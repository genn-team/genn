#pragma once

// Standard C++ includes
#include <string>

// Mini-parse includes
#include "statement.h"

//---------------------------------------------------------------------------
// MiniParse::PrettyPrinter
//---------------------------------------------------------------------------
namespace MiniParse::PrettyPrinter
{
std::string print(const Statement::StatementList &statements);
}