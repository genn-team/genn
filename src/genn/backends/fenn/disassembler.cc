#include "disassembler.h"

// Standard C++ includes
#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

// Standard C includes
#include <cassert>

// Common include
#include "isa.h"

using namespace GeNN::CodeGenerator::FeNN;

/*
    JALR    = 0b11001,
    JAL     = 0b11011,
    AUIPC   = 0b00101,
    SYSTEM  = 0b11100,*/
// Anonymous namespace
namespace
{
using DisassembleFunc = std::function<void(std::ostream&, uint32_t)>;

void disassembleLoad(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs1, funct3, rd] = decodeIType(inst);
    auto type = getLoadType(funct3);
    os << type._to_string() << " X" << rd << ", " << imm << "(X" << rs1 << ")";
}

void disassembleStore(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs2, rs1, funct3] = decodeSType(inst);
    const auto type = getStoreType(funct3);
    os << type._to_string() << " X" << rs2 << ", " << imm << "(X" << rs1 << ")";
}

void disassembleBranch(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs2, rs1, funct3] = decodeBType(inst);
    const auto type = getBranchType(funct3);
    os << type._to_string() << " X" << rs1 << ", X" << rs2 << ", " << imm;
}

void disassembleOP(std::ostream &os, uint32_t inst)
{
    const auto [funct7, rs2, rs1, funct3, rd] = decodeRType(inst);
    const auto type = getOpType(funct7, funct3);
    os << type._to_string() << " X" << rd << ", X" << rs1 << ", X" << rs2;
}

void disassembleOPImm(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs1, funct3, rd] = decodeIType(inst);
    const auto type = getOpImmType(imm, funct3);
    os << type._to_string() << " X" << rd << ", X" << rs1 << ", " << imm;
}

void disassembleLUI(std::ostream &os, uint32_t inst)
{
    const auto [imm, rd] = decodeUType(inst);
    os << "LUI X" << rd << ", " << imm;
}

void disassembleVOp(std::ostream &os, uint32_t inst)
{
    const auto [funct7, rs2, rs1, funct3, rd] = decodeRType(inst);
    const auto type = getVOpType(funct7, funct3);

    if(type == +VOpType::VMUL || type == +VOpType::VMUL_RN || type == +VOpType::VMUL_RS) {
        os << type._to_string() << " V" << rd << ", V" << rs1 << ", V" << rs2 << ", " << (funct7 & 0b1111);
    }
    else {
        os << type._to_string() << " V" << rd << ", V" << rs1 << ", V" << rs2;
    }
}

void disassembleVLUI(std::ostream &os, uint32_t inst)
{
    const auto [imm, rd] = decodeUType(inst);
    os << "VLUI V" << rd << ", " << imm;
}

void disassembleVTst(std::ostream &os, uint32_t inst)
{
    const auto [funct7, rs2, rs1, funct3, rd] = decodeRType(inst);
    const auto type = getVTstType(funct3);
    os << type._to_string() << " X" << rd << ", V" << rs1 << ", V" << rs2;
}

void disassembleVSel(std::ostream &os, uint32_t inst)
{
    const auto [funct7, rs2, rs1, funct3, rd] = decodeRType(inst);
    os << "VSEL V" << rd << ", X" << rs1 << ", V" << rs2;
}

void disassembleVLoad(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs1, funct3, rd] = decodeIType(inst);
    const auto type = getVLoadType(funct3);
    os << type._to_string() << " V" << rd << ", " << imm << "(X" << rs1 << ")";
}

void disassembleVStore(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs2, rs1, funct3] = decodeSType(inst);
    os << "VSTORE V" << rs2 << ", " << imm << "(X" << rs1 << ")";
}

void disassembleVMov(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs1, funct3, rd] = decodeIType(inst);
    const auto type = getVMovType(funct3);
    if(type == +VMovType::VFILL) {
        os << "VFILL V" << rd << ", X" << rs1;
    }
    else if(type == +VMovType::VEXTRACT) {
        os << "VEXTRACT X" << rd << ", V" << rs1 << ", " << imm;
    }
}

void disassembleVSpc(std::ostream &os, uint32_t inst)
{
    const auto [imm, rs1, funct3, rd] = decodeIType(inst);
    const auto type = getVSpcType(funct3);

    os << type._to_string() << " V" << rd;
}

const std::unordered_map<StandardOpCode, DisassembleFunc> standardInstructionDecoders{
    {StandardOpCode::LOAD, &disassembleLoad},
    {StandardOpCode::STORE, &disassembleStore},
    {StandardOpCode::BRANCH, &disassembleBranch},
    {StandardOpCode::OP, &disassembleOP},
    {StandardOpCode::OP_IMM, &disassembleOPImm},
    {StandardOpCode::LUI, &disassembleLUI},
};

const std::unordered_map<VectorOpCode, DisassembleFunc> vectorInstructionDecoders{
    {VectorOpCode::VSOP, &disassembleVOp},
    {VectorOpCode::VLUI, &disassembleVLUI},
    {VectorOpCode::VTST, &disassembleVTst},
    {VectorOpCode::VSEL, &disassembleVSel},
    {VectorOpCode::VLOAD, &disassembleVLoad},
    {VectorOpCode::VSTORE, &disassembleVStore},
    {VectorOpCode::VMOV, &disassembleVMov},
    {VectorOpCode::VSPC, &disassembleVSpc},
};
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::FeNN
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::FeNN 
{
void disassemble(std::ostream &os, uint32_t inst)
{
    // Extract 2-bit quadrant
    const uint32_t quadrant = inst & 0b11;
    const uint32_t opCode = (inst & 0b1111100) >> 2;

    // If instruction is in standard quadrant
    if(quadrant == standardQuadrant) {
        const auto i = standardInstructionDecoders.find(static_cast<StandardOpCode>(opCode));
        if(i != standardInstructionDecoders.cend()) {
            i->second(os, inst);
        }
        else {
            throw std::runtime_error("Unsupported standard instruction");
        }
    }
    // Otherwise, if it is from vector quadrant
    else if(quadrant == vectorQuadrant){
        const auto i = vectorInstructionDecoders.find(static_cast<VectorOpCode>(opCode));
        if(i != vectorInstructionDecoders.cend()) {
            i->second(os, inst);
        }
        else {
            throw std::runtime_error("Unsupported vector instruction");
        }
    }
    // Otherwise, throw
    else {
        throw std::runtime_error("Unsupported quadrant");
    }
}
}