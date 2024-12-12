#include "compiler.h"

// Third-party includes
#include <fast_float/fast_float.h>

// FeNN backend includes
#include "assembler.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::CodeGenerator::FeNN;
using namespace GeNN::CodeGenerator::FeNN::Compiler;
using namespace GeNN::Transpiler;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Statement::StatementList &statements, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
            std::optional<ScalarRegisterAllocator::RegisterPtr> maskRegister,
            ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
    :   m_Environment(environment), m_Context(context), m_MaskRegister(maskRegister), m_ResolvedTypes(resolvedTypes),
        m_ScalarRegisterAllocator(scalarRegisterAllocator), m_VectorRegisterAllocator(vectorRegisterAllocator)
    {
         for(auto &s : statements) {
            s.get()->accept(*this);
        }
    }

    Visitor(const Expression::ExpressionPtr &expression, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
            ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
    :   m_Environment(environment), m_Context(context), m_ResolvedTypes(resolvedTypes),
        m_ScalarRegisterAllocator(scalarRegisterAllocator), m_VectorRegisterAllocator(vectorRegisterAllocator)
    {
        expression.get()->accept(*this);
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    RegisterPtr getExpressionRegister() const
    {
        return m_ExpressionRegister.value().first;
    }

private:
    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        assert(false);
    }

    virtual void visit(const Expression::Assignment &assignement) final
    {
        const auto vecAssigneeReg = getExpressionVectorRegister(assignement.getAssignee());
        const auto vecValueReg = getExpressionVectorRegister(assignement.getValue());

        // If a mask is set
        // **TODO** only necessary when assigning to variables outside of masked scope
        const auto opType = assignement.getOperator().type;
        if(m_MaskRegister) {
            // If we're doing plain assignement, conditionally assign from value register directly
            if(opType == Token::Type::EQUAL) {
                m_Environment.get().getCodeGenerator().vsel(*vecAssigneeReg, *m_MaskRegister.value(), *vecValueReg);
            }
            // Otherwise
            else {
                // Generate assignement into temporary register
                // **TODO** if value reg is reusable, no real need for extra register
                const auto tempReg = m_VectorRegisterAllocator.getRegister();
                generateAssign(opType, *tempReg, *vecAssigneeReg, *vecValueReg);

                // Conditionally assign back to assignee register
                m_Environment.get().getCodeGenerator().vsel(*vecAssigneeReg, *m_MaskRegister.value(), *tempReg);
            }
        }
        // Otherwise, generate assignement directly into assignee register
        else {
            generateAssign(opType, *vecAssigneeReg, *vecAssigneeReg, *vecValueReg);
        }
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        const auto vecLeftReg = getExpressionVectorRegister(binary.getLeft());
        const auto vecRightReg = getExpressionVectorRegister(binary.getRight());

        // If operation is arithmetic
        const auto opType = binary.getOperator().type;
        if(opType == Token::Type::MINUS || opType == Token::Type::PLUS || opType == Token::Type::STAR) {
            const auto resultReg = m_VectorRegisterAllocator.getRegister();
            if(opType == Token::Type::MINUS) {
                // **TODO** saturation?
                m_Environment.get().getCodeGenerator().vsub(*resultReg, *vecLeftReg, *vecRightReg);
            }
            else if(opType == Token::Type::PLUS) {
                // **TODO** saturation?
                m_Environment.get().getCodeGenerator().vadd(*resultReg, *vecLeftReg, *vecRightReg);
            }
            else if(opType == Token::Type::STAR) {
                // **TODO** fixed point format and rounding
                m_Environment.get().getCodeGenerator().vmul(8, *resultReg, *vecLeftReg, *vecRightReg);
            }
            
            // Set result register
            // **NOTE** result is a temporary register so always re-usable
            setExpressionRegister(resultReg, true);
        }
        // Otherwise, if it is relational
        else if(opType == Token::Type::GREATER || opType == Token::Type::GREATER_EQUAL || opType == Token::Type::LESS
                || opType == Token::Type::LESS_EQUAL || opType == Token::Type::NOT_EQUAL || opType == Token::Type::EQUAL_EQUAL) 
        {
            const auto resultReg = m_ScalarRegisterAllocator.getRegister();

            if(opType == Token::Type::GREATER) {
                m_Environment.get().getCodeGenerator().vtlt(*resultReg, *vecRightReg, *vecLeftReg);
            }
            else if(opType == Token::Type::GREATER_EQUAL) {
                m_Environment.get().getCodeGenerator().vtge(*resultReg, *vecLeftReg, *vecRightReg);
            }
            else if(opType == Token::Type::LESS) {
                m_Environment.get().getCodeGenerator().vtlt(*resultReg, *vecLeftReg, *vecRightReg);
            }
            else if(opType == Token::Type::LESS_EQUAL) {
                m_Environment.get().getCodeGenerator().vtge(*resultReg, *vecRightReg, *vecLeftReg);
            }
            else if(opType == Token::Type::NOT_EQUAL) {
                m_Environment.get().getCodeGenerator().vtne(*resultReg, *vecLeftReg, *vecRightReg);
            }
            else if(opType == Token::Type::EQUAL_EQUAL) {
                m_Environment.get().getCodeGenerator().vteq(*resultReg, *vecLeftReg, *vecRightReg);
            }
            
            // Set result register
            // **NOTE** result is a temporary register so always re-usable
            setExpressionRegister(resultReg, true);
        }
        else {
            assert(false);
        }
    }

    virtual void visit(const Expression::Call &call) final
    {
        assert(false);
    }

    virtual void visit(const Expression::Cast &cast) final
    {
        // **NOTE** allow expression register to pass through
        cast.getExpression()->accept(*this);
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto conditionReg = getExpressionScalarRegister(conditional.getCondition());
        const auto trueReg = getExpressionVectorRegister(conditional.getTrue());
        const auto falseReg = getExpressionVectorRegister(conditional.getFalse());
        assert(isExpressionRegisterReusable());

        m_Environment.get().getCodeGenerator().vsel(*falseReg, *conditionReg, *trueReg);

        // Set result register
        setExpressionRegister(falseReg, true);
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        // **NOTE** allow expression register to pass through
        grouping.getExpression()->accept(*this);
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        const auto lexeme = literal.getValue().lexeme;
        const char *lexemeBegin = lexeme.c_str();
        const char *lexemeEnd = lexemeBegin + lexeme.size();
        
        // If literal is uint
        int64_t integerResult;
        if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            unsigned int result;
            auto answer = fast_float::from_chars(lexemeBegin, lexemeEnd, result);
            assert(answer.ec == std::errc());
            integerResult = result;
        }
        else if(literal.getValue().type == Token::Type::INT32_NUMBER) {
            int result;
            auto answer = fast_float::from_chars(lexemeBegin, lexemeEnd, result);
            assert(answer.ec == std::errc());
            integerResult = result;
        }
        else {
            // **TODO** fixed point
            float result;
            auto answer = fast_float::from_chars(lexemeBegin, lexemeEnd, result);
            assert(answer.ec == std::errc());
            integerResult = std::round(result * (1u << 8));
        }

        assert(integerResult >= std::numeric_limits<int16_t>::min());
        assert(integerResult <= std::numeric_limits<int16_t>::max());

        const auto resultReg = m_VectorRegisterAllocator.getRegister();
        m_Environment.get().getCodeGenerator().vlui(*resultReg, (uint32_t)integerResult);
        
        // Set result register
        // **NOTE** result is a temporary register so always re-usable
        setExpressionRegister(resultReg, true);
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        const auto scalarLeftReg = getExpressionScalarRegister(logical.getLeft());
        const auto scalarRightReg = getExpressionScalarRegister(logical.getRight());

        // **TODO** short-circuiting
        const auto resultReg = m_ScalarRegisterAllocator.getRegister();
        if(logical.getOperator().type == Token::Type::AMPERSAND_AMPERSAND) {
            m_Environment.get().getCodeGenerator().and_(*resultReg, *scalarLeftReg, *scalarRightReg);
        }
        else if(logical.getOperator().type == Token::Type::PIPE_PIPE){
            m_Environment.get().getCodeGenerator().or_(*resultReg, *scalarLeftReg, *scalarRightReg);
        }
        else {
            assert(false);
        }

        // Set result register
        // **NOTE** result is a temporary register so always re-usable
        setExpressionRegister(resultReg, true);
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        assert(false);
        //postfixIncDec.getTarget()->accept(*this);
        //m_Environment.get().getStream() <<  postfixIncDec.getOperator().lexeme;
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        assert(false);
        //m_Environment.get().getStream() << prefixIncDec.getOperator().lexeme;
        //prefixIncDec.getTarget()->accept(*this);
    }

    virtual void visit(const Expression::Identifier &variable) final
    {
        // Set result register
        // **NOTE** we don't want to re-use registers used for variables
        setExpressionRegister(m_Environment.get().getRegister(variable.getName().lexeme), false);
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        assert(false);
        //m_Environment.get().getStream() << unary.getOperator().lexeme;
        //unary.getRight()->accept(*this);
    }

    //---------------------------------------------------------------------------
    // Statement::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Statement::Break&) final
    {
        assert(false);
    }

    virtual void visit(const Statement::Compound &compound) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 

        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        for(auto &s : compound.getStatements()) {
            s->accept(*this);
        }

        // Restore old environment
        m_Environment = oldEnvironment;
    }

    virtual void visit(const Statement::Continue&) final
    {
        assert(false);
    }

    virtual void visit(const Statement::Do &doStatement) final
    {
        assert(false);
        /*m_Environment.get().getStream() << "do";
        doStatement.getBody()->accept(*this);
        m_Environment.get().getStream() << "while(";
        doStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ");" << std::endl;*/
    }

    virtual void visit(const Statement::Expression &expression) final
    {
        if(expression.getExpression()) {
            expression.getExpression()->accept(*this);
        }
    }

    virtual void visit(const Statement::For &forStatement) final
    {
        // Cache reference to current reference
        /*std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 

        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        m_Environment.get().getStream() << "for(";
        if(forStatement.getInitialiser()) {
            forStatement.getInitialiser()->accept(*this);
        }
        else {
            m_Environment.get().getStream() << ";";
        }
        m_Environment.get().getStream() << " ";

        if(forStatement.getCondition()) {
            forStatement.getCondition()->accept(*this);
        }

        m_Environment.get().getStream() << "; ";
        if(forStatement.getIncrement()) {
            forStatement.getIncrement()->accept(*this);
        }
        m_Environment.get().getStream() << ")";
        forStatement.getBody()->accept(*this);

        // Restore old environment
        m_Environment = oldEnvironment;*/
    }

    virtual void visit(const Statement::ForEachSynapse &forEachSynapseStatement) final
    {
        assert(false);
    }

    virtual void visit(const Statement::If &ifStatement) final
    {
        const auto scalarConditionReg = getExpressionScalarRegister(ifStatement.getCondition());

        // If we already have a mask register, AND it with result of condition into new register
        auto oldMaskRegister = m_MaskRegister;
        if(oldMaskRegister) {
            auto combinedMaskRegister = m_ScalarRegisterAllocator.getRegister();
            m_Environment.get().getCodeGenerator().and_(*combinedMaskRegister, *oldMaskRegister.value(), *scalarConditionReg);
            m_MaskRegister = combinedMaskRegister;
        }
        // Otherwise, just
        else {
            m_MaskRegister = scalarConditionReg;
        }

        ifStatement.getThenBranch()->accept(*this);


        if(ifStatement.getElseBranch()) {
            // Negate mask
            auto elseMaskRegister = m_ScalarRegisterAllocator.getRegister();
            m_Environment.get().getCodeGenerator().not_(*elseMaskRegister, *scalarConditionReg);

            // If we have an old mask, and it with this
            if(oldMaskRegister) {
                m_Environment.get().getCodeGenerator().and_(*elseMaskRegister, *oldMaskRegister.value(),
                                                            *elseMaskRegister);
            }

            m_MaskRegister = elseMaskRegister;

            ifStatement.getElseBranch()->accept(*this);
        }

        // Restore old mask register
        m_MaskRegister = oldMaskRegister;
    }

    virtual void visit(const Statement::Labelled &labelled) final
    {
        assert(false);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        assert(false);
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        const size_t numDeclarators = varDeclaration.getInitDeclaratorList().size();
        for(size_t i = 0; i < numDeclarators; i++) {
            const auto &var = varDeclaration.getInitDeclaratorList()[i];

            // If variable is assigned to something
            if(std::get<1>(var)) {
                // Evaluate value
                const auto valueVectorReg = getExpressionVectorRegister(std::get<1>(var).get());

                // If value register can be re-used, poach it for variable
                if(isExpressionRegisterReusable()) {
                    m_Environment.get().define(std::get<0>(var).lexeme, valueVectorReg);
                }
                else {
                    // Allocate register and define variable to it
                    const auto varReg = m_VectorRegisterAllocator.getRegister(std::get<0>(var).lexeme.c_str());
                    m_Environment.get().define(std::get<0>(var).lexeme, varReg);
                    
                    // Copy value into result
                    // **NOTE** mask never matters here as these variables are inherantly local
                    // **TODO** add flag alongside expression register to specify whether it can be trashed
                    m_Environment.get().getCodeGenerator().vadd(*varReg, *valueVectorReg,
                                                                *std::get<VectorRegisterAllocator::RegisterPtr>(m_Environment.get().getRegister("_zero")));
                }
            }
            // Otherwise, allocate register and define variable to it
            else {
                const auto varReg = m_VectorRegisterAllocator.getRegister(std::get<0>(var).lexeme.c_str());
                m_Environment.get().define(std::get<0>(var).lexeme, varReg);
            }
        }
    }

    virtual void visit(const Statement::While &whileStatement) final
    {
        assert(false);
        /*m_Environment.get().getStream() << "while(";
        whileStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ")" << std::endl;
        whileStatement.getBody()->accept(*this);*/
    }

    //---------------------------------------------------------------------------
    // Private methods
    //---------------------------------------------------------------------------
    bool isExpressionRegisterReusable() const
    {
        return m_ExpressionRegister.value().second;
    }

    void setExpressionRegister()
    {
        m_ExpressionRegister = std::nullopt;
    }

    void setExpressionRegister(RegisterPtr reg, bool reusable)
    {
        m_ExpressionRegister = std::make_pair(reg, reusable);
    }

    VectorRegisterAllocator::RegisterPtr getExpressionVectorRegister(const Expression::Base *expression)
    {
        expression->accept(*this);

        return std::get<VectorRegisterAllocator::RegisterPtr>(getExpressionRegister());
    }

    ScalarRegisterAllocator::RegisterPtr getExpressionScalarRegister(const Expression::Base *expression)
    {
        expression->accept(*this);

        return std::get<ScalarRegisterAllocator::RegisterPtr>(getExpressionRegister());
    }

    void generateAssign(Token::Type opType, VReg destinationReg, VReg assigneeReg, VReg valueReg)
    {
        if(opType == Token::Type::EQUAL) {
            m_Environment.get().getCodeGenerator().vadd(destinationReg, valueReg,
                                                        *std::get<VectorRegisterAllocator::RegisterPtr>(m_Environment.get().getRegister("_zero")));
        }
        else if(opType == Token::Type::STAR_EQUAL) {
            // **TODO** fixed point and rounding
            m_Environment.get().getCodeGenerator().vmul(8, destinationReg, assigneeReg, valueReg);
        }
        else if(opType == Token::Type::PLUS_EQUAL) {
            // **TODO** saturation
            m_Environment.get().getCodeGenerator().vadd(destinationReg, assigneeReg, valueReg);
        }
        else if(opType == Token::Type::MINUS_EQUAL) {
            // **TODO** saturation
            m_Environment.get().getCodeGenerator().vsub(destinationReg, assigneeReg, valueReg);
        }
        else {
            assert(false);
        }
    }
    
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentBase> m_Environment;
    const Type::TypeContext &m_Context;
    std::optional<std::pair<RegisterPtr, bool>> m_ExpressionRegister;

    std::optional<ScalarRegisterAllocator::RegisterPtr> m_MaskRegister;
    const TypeChecker::ResolvedTypeMap &m_ResolvedTypes;
    ScalarRegisterAllocator &m_ScalarRegisterAllocator;
    VectorRegisterAllocator &m_VectorRegisterAllocator;
};
}

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::FeNN::Compiler::EnvironmentInternal
//---------------------------------------------------------------------------
namespace GeNN::CodeGenerator::FeNN::Compiler
{
void EnvironmentInternal::define(const std::string &name, RegisterPtr reg)
{
    if(!m_LocalVariables.emplace(name, reg).second) {
        throw std::runtime_error("Redeclaration of variable");
    }
}
//----------------------------------------------------------------------------
RegisterPtr EnvironmentInternal::getRegister(const std::string &name)
{
    auto l = m_LocalVariables.find(name);
    if(l == m_LocalVariables.end()) {
        return m_Enclosing.getRegister(name);
    }
    else {
        return l->second;
    }
}
//----------------------------------------------------------------------------
Assembler::CodeGenerator &EnvironmentInternal::getCodeGenerator()
{
    return m_Enclosing.getCodeGenerator();
}
//----------------------------------------------------------------------------
void compile(const Statement::StatementList &statements, EnvironmentInternal &environment, 
             const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
             std::optional<ScalarRegisterAllocator::RegisterPtr> maskRegister, 
             ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
{
    Visitor visitor(statements, environment, context, resolvedTypes, maskRegister,
                    scalarRegisterAllocator, vectorRegisterAllocator);
}
//---------------------------------------------------------------------------
RegisterPtr compile(const Expression::ExpressionPtr &expression, EnvironmentInternal &environment,
                    const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
                    ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
{
    Visitor visitor(expression, environment, context, resolvedTypes,
                    scalarRegisterAllocator, vectorRegisterAllocator);
    return visitor.getExpressionRegister();
}
}