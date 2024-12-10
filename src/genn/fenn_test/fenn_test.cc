// Standard C++ includes
#include <variant>

// GeNN includes
#include "gennUtils.h"
#include "neuronModels.h"
#include "type.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"

// FeNN backend includes
#include "assembler.h"
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

}