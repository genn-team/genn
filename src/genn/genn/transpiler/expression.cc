#include "expression.h"

#define IMPLEMENT_ACCEPT(CLASS_NAME)                                        \
    void MiniParse::Expression::CLASS_NAME::accept(Visitor &visitor) const  \
    {                                                                       \
        visitor.visit(*this);                                               \
    }


IMPLEMENT_ACCEPT(ArraySubscript)
IMPLEMENT_ACCEPT(Assignment)
IMPLEMENT_ACCEPT(Binary)
IMPLEMENT_ACCEPT(Call)
IMPLEMENT_ACCEPT(Cast)
IMPLEMENT_ACCEPT(Conditional)
IMPLEMENT_ACCEPT(Grouping)
IMPLEMENT_ACCEPT(Literal)
IMPLEMENT_ACCEPT(Logical)
IMPLEMENT_ACCEPT(PrefixIncDec)
IMPLEMENT_ACCEPT(PostfixIncDec)
IMPLEMENT_ACCEPT(Variable)
IMPLEMENT_ACCEPT(Unary)