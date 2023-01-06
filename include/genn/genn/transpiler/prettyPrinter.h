#pragma once

// Standard C++ includes
#include <string>

// Transpiler includes
#include "transpiler/statement.h"

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::PrettyPrinter
{
std::string print(const Statement::StatementList &statements);
}