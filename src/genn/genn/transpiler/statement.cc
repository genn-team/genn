#include "statement.h"

#define IMPLEMENT_ACCEPT(CLASS_NAME)                                        \
    void MiniParse::Statement::CLASS_NAME::accept(Visitor &visitor) const   \
    {                                                                       \
        visitor.visit(*this);                                               \
    }

// Implement accept methods
IMPLEMENT_ACCEPT(Break)
IMPLEMENT_ACCEPT(Compound)
IMPLEMENT_ACCEPT(Continue)
IMPLEMENT_ACCEPT(Do)
IMPLEMENT_ACCEPT(Expression)
IMPLEMENT_ACCEPT(For)
IMPLEMENT_ACCEPT(If)
IMPLEMENT_ACCEPT(Labelled)
IMPLEMENT_ACCEPT(Switch)
IMPLEMENT_ACCEPT(VarDeclaration)
IMPLEMENT_ACCEPT(While)
IMPLEMENT_ACCEPT(Print)