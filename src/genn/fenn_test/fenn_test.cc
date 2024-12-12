// Standard C++ includes
#include <variant>

// Third-party includes
#include <fast_float/fast_float.h>
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "neuronModels.h"
#include "type.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"

// FeNN backend includes
#include "assembler.h"
#include "disassembler.h"
#include "registerAllocator.h"

using namespace GeNN;
using namespace GeNN::Transpiler;
using namespace GeNN::CodeGenerator::FeNN;

using RegisterPtr = std::variant<ScalarRegisterAllocator::RegisterPtr, VectorRegisterAllocator::RegisterPtr>;

// EnvironmentBase
//! Will live in Assembler
class EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Define identifier as corresponding register
    virtual void define(const std::string &name, RegisterPtr reg) = 0;
    
    //! Get the register to use for the named identifier
    virtual RegisterPtr getRegister(const std::string &name) = 0;
    
    //! Get stream to write code within this environment to
    virtual Assembler::CodeGenerator &getCodeGenerator() = 0;

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    RegisterPtr operator[] (const std::string &name)
    {
        return getRegister(name);
    }
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter::EnvironmentInternal
//---------------------------------------------------------------------------
class EnvironmentInternal : public EnvironmentBase
{
public:
    EnvironmentInternal(EnvironmentBase &enclosing)
    :   m_Enclosing(enclosing)
    {
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual void define(const std::string &name, RegisterPtr reg) final
    {
        if(!m_LocalVariables.emplace(name, reg).second) {
            throw std::runtime_error("Redeclaration of variable");
        }
    }

    virtual RegisterPtr getRegister(const std::string &name) final
    {
        auto l = m_LocalVariables.find(name);
        if(l == m_LocalVariables.end()) {
            return m_Enclosing.getRegister(name);
        }
        else {
            return l->second;
        }
    }

    virtual Assembler::CodeGenerator &getCodeGenerator() final
    {
        return m_Enclosing.getCodeGenerator();
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_map<std::string, RegisterPtr> m_LocalVariables;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Statement::StatementList &statements, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
            ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
    :   m_Environment(environment), m_Context(context), m_ResolvedTypes(resolvedTypes), 
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
        assignement.getAssignee()->accept(*this);
        const auto assigneeReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(assigneeReg));

        assignement.getValue()->accept(*this);
        const auto valueReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(valueReg));

        // If a mask is set, use vsel to conditionally assign
        // **TODO** only necessary when assigning to variables outside of masked scope
        if(m_MaskRegister) {
             m_Environment.get().getCodeGenerator().vsel(*std::get<VectorRegisterAllocator::RegisterPtr>(assigneeReg), *m_MaskRegister.value(), 
                                                         *std::get<VectorRegisterAllocator::RegisterPtr>(valueReg));
        }
        // Otherwise, copy assigneeReg into result
        // **TODO** add flag alongside expression register to specify whether it can be trashed
        else {
            m_Environment.get().getCodeGenerator().vadd(*std::get<VectorRegisterAllocator::RegisterPtr>(assigneeReg), *std::get<VectorRegisterAllocator::RegisterPtr>(valueReg),
                                                        *std::get<VectorRegisterAllocator::RegisterPtr>(m_Environment.get().getRegister("_zero")));
        }

    }

    virtual void visit(const Expression::Binary &binary) final
    {
        binary.getLeft()->accept(*this);
        const auto leftReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(leftReg));

        binary.getRight()->accept(*this);
        const auto rightReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(rightReg));

        // If operation is arithmetic
        const auto opType = binary.getOperator().type;
        if(opType == Token::Type::MINUS || opType == Token::Type::PLUS || opType == Token::Type::STAR) {
            const auto resultReg = m_VectorRegisterAllocator.getRegister();
            if(opType == Token::Type::MINUS) {
                // **TODO** saturation?
                m_Environment.get().getCodeGenerator().vsub(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg), 
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            else if(opType == Token::Type::PLUS) {
                // **TODO** saturation?
                m_Environment.get().getCodeGenerator().vadd(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg), 
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            else if(opType == Token::Type::STAR) {
                // **TODO** fixed point format
                m_Environment.get().getCodeGenerator().vmul(8, *resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg), 
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            
            // Set result register
            m_ExpressionRegister = resultReg;
        }
        // Otherwise, if it is relational
        else if(opType == Token::Type::GREATER || opType == Token::Type::GREATER_EQUAL || opType == Token::Type::LESS
                || opType == Token::Type::LESS_EQUAL || opType == Token::Type::NOT_EQUAL || opType == Token::Type::EQUAL_EQUAL) 
        {
            const auto resultReg = m_ScalarRegisterAllocator.getRegister();

            if(opType == Token::Type::GREATER) {
                m_Environment.get().getCodeGenerator().vtlt(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg));
            }
            else if(opType == Token::Type::GREATER_EQUAL) {
                m_Environment.get().getCodeGenerator().vtge(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            else if(opType == Token::Type::LESS) {
                m_Environment.get().getCodeGenerator().vtlt(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            else if(opType == Token::Type::LESS_EQUAL) {
                m_Environment.get().getCodeGenerator().vtge(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg));
            }
            else if(opType == Token::Type::NOT_EQUAL) {
                m_Environment.get().getCodeGenerator().vtne(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            else if(opType == Token::Type::EQUAL_EQUAL) {
                m_Environment.get().getCodeGenerator().vteq(*resultReg, *std::get<VectorRegisterAllocator::RegisterPtr>(leftReg),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(rightReg));
            }
            // Set result register
            m_ExpressionRegister = resultReg;
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
        // **NOTE** leave expression register alone
        cast.getExpression()->accept(*this);
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        conditional.getCondition()->accept(*this);
        const auto conditionReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<ScalarRegisterAllocator::RegisterPtr>(conditionReg));

        conditional.getTrue()->accept(*this);
        const auto trueReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(trueReg));

        conditional.getFalse()->accept(*this);
        const auto falseReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<VectorRegisterAllocator::RegisterPtr>(falseReg));

        m_Environment.get().getCodeGenerator().vsel(*std::get<VectorRegisterAllocator::RegisterPtr>(falseReg),
                                                    *std::get<ScalarRegisterAllocator::RegisterPtr>(conditionReg),
                                                    *std::get<VectorRegisterAllocator::RegisterPtr>(trueReg));
        m_ExpressionRegister = falseReg;
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        // **NOTE** leave expression register alone
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
        m_ExpressionRegister = resultReg;
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        const auto leftReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<ScalarRegisterAllocator::RegisterPtr>(leftReg));

        logical.getRight()->accept(*this);
        const auto rightReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<ScalarRegisterAllocator::RegisterPtr>(rightReg));

        // **TODO** short-circuiting
        const auto resultReg = m_ScalarRegisterAllocator.getRegister();
        if(logical.getOperator().type == Token::Type::AMPERSAND_AMPERSAND) {
            m_Environment.get().getCodeGenerator().and_(*resultReg, *std::get<ScalarRegisterAllocator::RegisterPtr>(leftReg),
                                                        *std::get<ScalarRegisterAllocator::RegisterPtr>(rightReg));
        }
        else if(logical.getOperator().type == Token::Type::PIPE_PIPE){
            m_Environment.get().getCodeGenerator().or_(*resultReg, *std::get<ScalarRegisterAllocator::RegisterPtr>(leftReg),
                                                       *std::get<ScalarRegisterAllocator::RegisterPtr>(rightReg));
        }
        else {
            assert(false);
        }

        m_ExpressionRegister = resultReg;
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
        m_ExpressionRegister = m_Environment.get().getRegister(variable.getName().lexeme);
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
        ifStatement.getCondition()->accept(*this);
        const auto conditionReg = m_ExpressionRegister.value();
        assert(std::holds_alternative<ScalarRegisterAllocator::RegisterPtr>(conditionReg));

        // If we already have a mask register, AND it with result of condition into new register
        auto oldMaskRegister = m_MaskRegister;
        if(oldMaskRegister) {
            auto combinedMaskRegister = m_ScalarRegisterAllocator.getRegister();
            m_Environment.get().getCodeGenerator().and_(*combinedMaskRegister, *oldMaskRegister.value(),
                                                        *std::get<ScalarRegisterAllocator::RegisterPtr>(conditionReg));
            m_MaskRegister = combinedMaskRegister;
        }
        // Otherwise, just
        else {
            m_MaskRegister = std::get<ScalarRegisterAllocator::RegisterPtr>(conditionReg);
        }

        ifStatement.getThenBranch()->accept(*this);


        if(ifStatement.getElseBranch()) {
            // Negate mask
            auto elseMaskRegister = m_ScalarRegisterAllocator.getRegister();
            m_Environment.get().getCodeGenerator().not_(*elseMaskRegister, *std::get<ScalarRegisterAllocator::RegisterPtr>(conditionReg));

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

            // Allocate register and define variable to it
            const auto varReg = m_VectorRegisterAllocator.getRegister(std::get<0>(var).lexeme.c_str());
            m_Environment.get().define(std::get<0>(var).lexeme, varReg);
            
            if(std::get<1>(var)) {
                std::get<1>(var)->accept(*this);

                // Copy result into result
                // **NOTE** mask never matters here as these variables are inherantly local
                // **TODO** add flag alongside expression register to specify whether it can be trashed
                m_Environment.get().getCodeGenerator().vadd(*varReg, *std::get<VectorRegisterAllocator::RegisterPtr>(*m_ExpressionRegister),
                                                            *std::get<VectorRegisterAllocator::RegisterPtr>(m_Environment.get().getRegister("_zero")));
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

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentBase> m_Environment;
    const Type::TypeContext &m_Context;
    std::optional<RegisterPtr> m_ExpressionRegister;
    std::optional<ScalarRegisterAllocator::RegisterPtr> m_MaskRegister;
    const TypeChecker::ResolvedTypeMap &m_ResolvedTypes;
    ScalarRegisterAllocator &m_ScalarRegisterAllocator;
    VectorRegisterAllocator &m_VectorRegisterAllocator;
};

class EnvironmentExternal : public EnvironmentBase, public Transpiler::TypeChecker::EnvironmentBase
{
public:
    explicit EnvironmentExternal(EnvironmentExternal &enclosing)
    :   m_Context{&enclosing, &enclosing, nullptr}
    {
    }

    explicit EnvironmentExternal(::EnvironmentBase &enclosing)
    :   m_Context{nullptr, &enclosing, nullptr}
    {
    }

    explicit EnvironmentExternal(Assembler::CodeGenerator &os)
    :   m_Context{nullptr, nullptr, &os}
    {
    }


    EnvironmentExternal(const EnvironmentExternal&) = delete;

    //------------------------------------------------------------------------
    // Assembler::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual void define(const std::string &name, RegisterPtr reg) override
    {
        throw std::runtime_error("Cannot declare variable in external environment");
    }

    virtual RegisterPtr getRegister(const std::string &name) final
    {
        // If name isn't found in environment
        auto env = m_Environment.find(name);
        if (env == m_Environment.end()) {
            // If context includes a pretty-printing environment, get name from it
            if(std::get<1>(m_Context)) {
                return std::get<1>(m_Context)->getRegister(name);
            }
            // Otherwise, give error
            else {
                throw std::runtime_error("Identifier '" + name + "' undefined"); 
            }
        }
        // Otherwise, get name from payload
        else {
            return std::get<1>(env->second);
        }
    }

    //! Get stream to write code within this environment to
    virtual Assembler::CodeGenerator &getCodeGenerator() final
    {
        // If context includes a code stream
        if(std::get<2>(m_Context)) {
            return *std::get<2>(m_Context);
        }
        // Otherwise
        else {
            // Assert that there is a pretty printing environment
            assert(std::get<1>(m_Context));

            // Return its stream
            return std::get<1>(m_Context)->getCodeGenerator();
        }
    }
   
    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual void define(const Transpiler::Token &name, const GeNN::Type::ResolvedType &type, 
                        Transpiler::ErrorHandlerBase &errorHandler) override
    {
        throw std::runtime_error("Cannot declare variable in external environment");
    }

    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final
    {
        // If name isn't found in environment
        auto env = m_Environment.find(name.lexeme);
        if (env == m_Environment.end()) {
            // If context includes a type-checking environment, get type from it
            if(std::get<0>(m_Context)) {
                return std::get<0>(m_Context)->getTypes(name, errorHandler); 
            }
            // Otherwise, give error
            else {
                errorHandler.error(name, "Undefined identifier");
                throw TypeChecker::TypeCheckError();
            }
        }
        // Otherwise, return type of variables
        else {
            return {std::get<0>(env->second)};
        }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type (for type-checking) and a value (for pretty-printing) to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, RegisterPtr reg)
    {
         if(!m_Environment.try_emplace(name, type, reg).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }
  

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::tuple<Transpiler::TypeChecker::EnvironmentBase*, ::EnvironmentBase*, Assembler::CodeGenerator*> m_Context;
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, RegisterPtr>> m_Environment;
};


int main()
{
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    Logging::init(plog::Severity::info, plog::Severity::info, plog::Severity::debug, plog::Severity::info,
                  &consoleAppender, &consoleAppender, &consoleAppender, &consoleAppender);

    // Initialise backend logging
    plog::init<Logging::CHANNEL_BACKEND>(plog::Severity::debug, &consoleAppender);
    

    const Type::TypeContext typeContext = {{"scalar", Type::Int16}}; 
    const auto *model = NeuronModels::LIF::getInstance();

    // Scan model code strings
    const auto simCodeTokens = Utils::scanCode(model->getSimCode(), "sim code");
    const auto resetCodeTokens = Utils::scanCode(model->getResetCode(), "reset code");
    const auto thresholdCodeTokens = Utils::scanCode(model->getThresholdConditionCode(), "threshold condition code");

    // Parse model code strings
    ErrorHandler errorHandler("Errors");
    const auto simCodeStatements = Parser::parseBlockItemList(simCodeTokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }
    const auto resetCodeStatements = Parser::parseBlockItemList(resetCodeTokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }
    const auto thresholdCodeExpression = Parser::parseExpression(thresholdCodeTokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }

    ScalarRegisterAllocator scalarRegisterAllocator;
    VectorRegisterAllocator vectorRegisterAllocator;
    Assembler::CodeGenerator codeGenerator;
    EnvironmentExternal env(codeGenerator);

    // Allocate registers for Isyn and dt
    env.add(Type::Int16, "Isyn", vectorRegisterAllocator.getRegister("Isyn V"));
    env.add(Type::Int16, "dt", vectorRegisterAllocator.getRegister("dt V"));

    env.add(Type::Int16, "_zero", vectorRegisterAllocator.getRegister("_zero V"));

    // Allocate registers for neuron variables
    const auto neuronVars = model->getVars();
    for(const auto &v : neuronVars) {
        env.add(Type::Int16, v.name, vectorRegisterAllocator.getRegister((v.name + " V").c_str()));
    }

    // Allocate registers for neuron parameters
    const auto neuronParams = model->getParams();
    for(const auto &p : neuronParams) {
        env.add(Type::Int16, p.name, vectorRegisterAllocator.getRegister((p.name + " V").c_str()));
    }

    // Allocate registers for neuron derived
    const auto neuronDerivedParams = model->getDerivedParams();
    for(const auto &d : neuronDerivedParams) {
        env.add(Type::Int16, d.name, vectorRegisterAllocator.getRegister((d.name + " V").c_str()));
    }

    // Resolve types within one scope
    TypeChecker::EnvironmentInternal typeCheckEnv(env);
    EnvironmentInternal assemblerEnv(env);
    {
        const auto resolvedTypeMap = TypeChecker::typeCheck(simCodeStatements, typeCheckEnv, typeContext, errorHandler);
        if(errorHandler.hasError()) {
            throw std::runtime_error("Type check error " + errorHandler.getContext());
        }


        Visitor visitor(simCodeStatements, assemblerEnv, typeContext, 
                        resolvedTypeMap, scalarRegisterAllocator, vectorRegisterAllocator);
    }

    for(uint32_t i: codeGenerator.getCode()) {
        disassemble(std::cout, i);
        std::cout << std::endl;
    }
}