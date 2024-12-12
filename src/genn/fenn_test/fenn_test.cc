// Standard C++ includes
#include <variant>

// Third-party includes
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
#include "compiler.h"
#include "disassembler.h"
#include "registerAllocator.h"

using namespace GeNN;
using namespace GeNN::Transpiler;
using namespace GeNN::CodeGenerator::FeNN;

namespace
{

Compiler::RegisterPtr compileExpression(const std::vector<Token> &tokens, const Type::TypeContext &typeContext, 
                                        TypeChecker::EnvironmentInternal &typeCheckEnv, Compiler::EnvironmentInternal &compilerEnv,
                                        ErrorHandler &errorHandler, ScalarRegisterAllocator &scalarRegisterAllocator, 
                                        VectorRegisterAllocator &vectorRegisterAllocator)
{
    // Parse tokens as expression
    auto expression = Parser::parseExpression(tokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }

    // Resolve types
    auto resolvedTypes = TypeChecker::typeCheck(expression.get(), typeCheckEnv, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Type check error " + errorHandler.getContext());
    }

    // Compile
    return Compiler::compile(expression, compilerEnv, typeContext, resolvedTypes,
                             scalarRegisterAllocator, vectorRegisterAllocator);
}

void compileStatements(const std::vector<Token> &tokens, const Type::TypeContext &typeContext,
                       TypeChecker::EnvironmentInternal &typeCheckEnv, Compiler::EnvironmentInternal &compilerEnv,
                       ErrorHandler &errorHandler, Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                       std::optional<ScalarRegisterAllocator::RegisterPtr> maskRegister, 
                       ScalarRegisterAllocator &scalarRegisterAllocator, VectorRegisterAllocator &vectorRegisterAllocator)
{
    using namespace Transpiler;

    // Parse tokens as block item list (function body)
    auto updateStatements = Parser::parseBlockItemList(tokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }

    // Resolve types
    auto resolvedTypes = TypeChecker::typeCheck(updateStatements, typeCheckEnv, typeContext, 
                                                errorHandler, forEachSynapseTypeCheckHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Type check error " + errorHandler.getContext());
    }

    // Compile
    Compiler::compile(updateStatements, compilerEnv, typeContext, resolvedTypes,
                      maskRegister, scalarRegisterAllocator, vectorRegisterAllocator);
}

class EnvironmentExternal : public Compiler::EnvironmentBase, public Transpiler::TypeChecker::EnvironmentBase
{
public:
    explicit EnvironmentExternal(EnvironmentExternal &enclosing)
    :   m_Context{&enclosing, &enclosing, nullptr}
    {
    }

    explicit EnvironmentExternal(Compiler::EnvironmentBase &enclosing)
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
    virtual void define(const std::string &name, Compiler::RegisterPtr reg) override
    {
        throw std::runtime_error("Cannot declare variable in external environment");
    }

    virtual Compiler::RegisterPtr getRegister(const std::string &name) final
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
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, Compiler::RegisterPtr reg)
    {
         if(!m_Environment.try_emplace(name, type, reg).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }
  

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::tuple<Transpiler::TypeChecker::EnvironmentBase*, Compiler::EnvironmentBase*, Assembler::CodeGenerator*> m_Context;
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, Compiler::RegisterPtr>> m_Environment;
};
}

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
    Compiler::EnvironmentInternal compilerEnv(env);
    ErrorHandler errorHandler("Errors");
    
    // Compile sim code
    compileStatements(simCodeTokens, typeContext, typeCheckEnv, compilerEnv,
                      errorHandler, nullptr, std::nullopt, scalarRegisterAllocator,
                      vectorRegisterAllocator);

    // Compile spike expression
    const auto spikeReg = std::get<ScalarRegisterAllocator::RegisterPtr>(
        compileExpression(thresholdCodeTokens, typeContext, typeCheckEnv,
                          compilerEnv, errorHandler, scalarRegisterAllocator,
                          vectorRegisterAllocator));
    
    // Compile reset code
    compileStatements(resetCodeTokens, typeContext, typeCheckEnv, compilerEnv,
                      errorHandler, nullptr, spikeReg, scalarRegisterAllocator,
                      vectorRegisterAllocator);

    for(uint32_t i: codeGenerator.getCode()) {
        disassemble(std::cout, i);
        std::cout << std::endl;
    }
}