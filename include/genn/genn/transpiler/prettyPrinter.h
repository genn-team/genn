#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/statement.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class CodeStream;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::PrettyPrinter
{
void print(CodeGenerator::CodeStream &os, const Statement::StatementList &statements, const Type::TypeContext &context);
}
