#pragma once
/*!
    @file xbyak_riscv.hpp
    @brief Xbyak_riscv ; JIT assembler for RISC-V
    @author herumi
    @url https://github.com/herumi/xbyak_riscv
    @note modified new BSD license
    http://opensource.org/licenses/BSD-3-Clause
*/

// Copyright (C), 2023, KNS Group LLC (YADRO)

// Standard C++ includes
#include <iostream>
#include <iomanip>
#include <list>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

// Standard C includes
#include <cassert>
#include <cstdint>

// RISC-V common includes
#include "common/isa.h"

// Forward declarations
class CodeGenerator;

enum {
    ERR_NONE,
    ERR_OFFSET_IS_TOO_BIG,
    ERR_CODE_IS_TOO_BIG,
    ERR_IMM_IS_TOO_BIG,
    ERR_INVALID_IMM_OF_JAL,
    ERR_INVALID_IMM_OF_BTYPE,
    ERR_LABEL_IS_NOT_FOUND,
    ERR_LABEL_IS_REDEFINED,
    ERR_LABEL_IS_TOO_FAR,
    ERR_LABEL_IS_NOT_SET_BY_L,
    ERR_LABEL_IS_ALREADY_SET_BY_L,
    ERR_CANT_PROTECT,
    ERR_CANT_ALLOC,
    ERR_BAD_PARAMETER,
    ERR_MUNMAP,
    ERR_INTERNAL,
    ERR_MAX,
};

//----------------------------------------------------------------------------
// Error
//----------------------------------------------------------------------------
class Error : public std::exception {
public:
    explicit Error(int err);
    operator int() const { return err_; }
    const char *what() const noexcept;

private:
    int err_;
};

//----------------------------------------------------------------------------
// Label
//----------------------------------------------------------------------------
class Label 
{
public:
    Label() : cg(nullptr), id(0) {}
    Label(const Label& rhs);
    Label& operator=(const Label& rhs);
    ~Label();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    uint32_t getAddress() const;

private:
    friend CodeGenerator;

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void clear() { cg = nullptr; id = 0; }
    int getId() const { return id; }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    mutable CodeGenerator *cg;
    mutable int id;
};

class CodeGenerator
{
public:
    // constructor
    CodeGenerator()
    {
        reset();
    }

     ~CodeGenerator()
    {
        resetLabelPtrList();
    }

    CodeGenerator operator=(const CodeGenerator&) = delete;

    void L(Label& label) { defineClabel(label); }
    Label L() { Label label; L(label); return label; }
  
    bool hasUndefinedLabel() const { return hasUndefClabel(); }
    const auto &getCode() const{ return code_; }

    void add(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 0, 0x0, rd, rs1, rs2); }
    void sub(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 0, 0x20, rd, rs1, rs2); }
    void sll(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 1, 0x0, rd, rs1, rs2); }
    void slt(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 2, 0x0, rd, rs1, rs2); }
    void sltu(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 3, 0x0, rd, rs1, rs2); }
    void xor_(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 4, 0x0, rd, rs1, rs2); }
    void srl(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 5, 0x0, rd, rs1, rs2); }
    void sra(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 5, 0x20, rd, rs1, rs2); }
    void or_(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 6, 0x0, rd, rs1, rs2); }
    void and_(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 7, 0x0, rd, rs1, rs2); }
    void mul(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 0, 0x1, rd, rs1, rs2); }
    void mulh(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 1, 0x1, rd, rs1, rs2); }
    void mulhsu(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 2, 0x1, rd, rs1, rs2); }
    void mulhu(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 3, 0x1, rd, rs1, rs2); }
    void div(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 4, 0x1, rd, rs1, rs2); }
    void divu(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 5, 0x1, rd, rs1, rs2); }
    void rem(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 6, 0x1, rd, rs1, rs2); }
    void remu(Reg rd, Reg rs1, Reg rs2) { Rtype(StandardOpCode::OP, 7, 0x1, rd, rs1, rs2); }
    void addi(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 0, rd, rs1, imm); }
    void slti(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 2, rd, rs1, imm); }
    void sltiu(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 3, rd, rs1, imm); }
    void xori(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 4, rd, rs1, imm); }
    void ori(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 6, rd, rs1, imm); }
    void andi(Reg rd, Reg rs1, int imm) { Itype(StandardOpCode::OP_IMM, 7, rd, rs1, imm); }
    // load-op rd, imm(addr); rd = addr[imm];
    void jalr(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::JALR, 0, rd, addr, imm); }
    void lb(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 0, rd, addr, imm); }
    void lh(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 1, rd, addr, imm); }
    void lw(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 2, rd, addr, imm); }
    void lbu(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 4, rd, addr, imm); }
    void lhu(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 5, rd, addr, imm); }
    void lwu(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 6, rd, addr, imm); }
    void ld(Reg rd, Reg addr, int imm = 0) { Itype(StandardOpCode::LOAD, 3, rd, addr, imm); }
    void auipc(Reg rd, uint32_t imm) { Utype(StandardOpCode::AUIPC, rd, imm); }
    void lui(Reg rd, uint32_t imm) { Utype(StandardOpCode::LUI, rd, imm); }
    void slli(Reg rd, Reg rs1, uint32_t shamt) { opShift(0x0, 1, StandardOpCode::OP_IMM, rd, rs1, shamt); }
    void srli(Reg rd, Reg rs1, uint32_t shamt) { opShift(0x0, 5, StandardOpCode::OP_IMM, rd, rs1, shamt); }
    void srai(Reg rd, Reg rs1, uint32_t shamt) { opShift(0x20, 5, StandardOpCode::OP_IMM, rd, rs1, shamt); }
    void ecall() { append4B(0x73); }
    void ebreak() { append4B(0x00100073); }
    // store-op rs, imm(addr) ; addr[imm] = rs;
    void sb(Reg rs, Reg addr, int imm = 0) { Stype(StandardOpCode::STORE, 0, addr, rs, imm); }
    void sh(Reg rs, Reg addr, int imm = 0) { Stype(StandardOpCode::STORE, 1, addr, rs, imm); }
    void sw(Reg rs, Reg addr, int imm = 0) { Stype(StandardOpCode::STORE, 2, addr, rs, imm); }
    void sd(Reg rs, Reg addr, int imm = 0) { Stype(StandardOpCode::STORE, 3, addr, rs, imm); }
    void beq(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 0, rs1, rs2); opJmp(label, jmp); }
    void bne(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 1, rs1, rs2); opJmp(label, jmp); }
    void blt(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 4, rs1, rs2); opJmp(label, jmp); }
    void bge(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 5, rs1, rs2); opJmp(label, jmp); }
    void bltu(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 6, rs1, rs2); opJmp(label, jmp); }
    void bgeu(Reg rs1, Reg rs2, const Label& label) { Jmp jmp(getCurr(), StandardOpCode::BRANCH, 7, rs1, rs2); opJmp(label, jmp); }
    void beqz(Reg rs, const Label& label) { beq(rs, Reg::X0, label); }
    void bnez(Reg rs, const Label& label) { bne(rs, Reg::X0, label); }
    void blez(Reg rs, const Label& label) { bge(Reg::X0, rs, label); }
    void bgez(Reg rs, const Label& label) { bge(rs, Reg::X0, label); }
    void bltz(Reg rs, const Label& label) { blt(rs, Reg::X0, label); }
    void bgtz(Reg rs, const Label& label) { blt(Reg::X0, rs, label); }
    void bgt(Reg rs, Reg rt, const Label& label) { blt(rt, rs, label); }
    void ble(Reg rs, Reg rt, const Label& label) { bge(rt, rs, label); }
    void bgtu(Reg rs, Reg rt, const Label& label) { bltu(rt, rs, label); }
    void bleu(Reg rs, Reg rt, const Label& label) { bgeu(rt, rs, label); }
    void csrrw(Reg rd, CSR csr, Reg rs1) { opCSR(0x1073, csr, rs1, rd); }
    void csrrs(Reg rd, CSR csr, Reg rs1) { opCSR(0x2073, csr, rs1, rd); }
    void csrrc(Reg rd, CSR csr, Reg rs1) { opCSR(0x3073, csr, rs1, rd); }
    void csrrwi(Reg rd, CSR csr, uint32_t imm) { opCSR(0x5073, csr, imm, rd); }
    void csrrsi(Reg rd, CSR csr, uint32_t imm) { opCSR(0x6073, csr, imm, rd); }
    void csrrci(Reg rd, CSR csr, uint32_t imm) { opCSR(0x7073, csr, imm, rd); }

    void nop() { addi(Reg::X0, Reg::X0, 0); }
    void li(Reg rd, int imm);
    void mv(Reg rd, Reg rs) { addi(rd, rs, 0); }
    void not_(Reg rd, Reg rs) { xori(rd, rs, -1); }
    void neg(Reg rd, Reg rs) { sub(rd, Reg::X0, rs); }
    void sext_b(Reg rd, Reg rs) { slli(rd, rs, 24); srai(rd, rd, 24); }
    void sext_h(Reg rd, Reg rs) { slli(rd, rs, 16); srai(rd, rd, 16); }
    void zext_b(Reg rd, Reg rs) { andi(rd, rs, 255); }
    void zext_h(Reg rd, Reg rs) { slli(rd, rs, 16); srli(rd, rd, 16); }
    void zext_w(Reg rd, Reg rs) { slli(rd, rs, 0); srli(rd, rd, 0); }
    void seqz(Reg rd, Reg rs) { sltiu(rd, rs, 1); }
    void snez(Reg rd, Reg rs) { sltu(rd, Reg::X0, rs); }
    void sltz(Reg rd, Reg rs) { slt(rd, rs, Reg::X0); }
    void sgtz(Reg rd, Reg rs) { slt(rd, Reg::X0, rs); }
    void fence() { append4B(0x0ff0000f); }
    void j_(const Label& label) { jal(Reg::X0, label); }
    void jal(Reg rd, const Label& label) { Jmp jmp(getCurr(), 0x6f, rd); opJmp(label, jmp); }
    void jr(Reg rs) { jalr(Reg::X0, rs, 0); }
    void jalr(Reg rs) { jalr(Reg::X1, rs, 0); }
    void ret() { jalr(Reg::X0, Reg::X1); }

    void csrr(Reg rd, CSR csr) { csrrs(rd, csr, Reg::X0); }
    void csrw(CSR csr, Reg rs) { csrrw(Reg::X0, csr, rs); }
    void csrs(CSR csr, Reg rs) { csrrs(Reg::X0, csr, rs); }
    void csrc(CSR csr, Reg rs) { csrrc(Reg::X0, csr, rs); }
    void csrwi(CSR csr, uint32_t imm) { csrrwi(Reg::X0, csr, imm); }
    void csrsi(CSR csr, uint32_t imm) { csrrsi(Reg::X0, csr, imm); }
    void csrci(CSR csr, uint32_t imm) { csrrci(Reg::X0, csr, imm); }

    // Zbb instructions
    void clz(Reg rd, Reg rs) { opShift(0x30, 1, StandardOpCode::OP_IMM, rd, rs, 0); }
    void ctz(Reg rd, Reg rs) { opShift(0x30, 1, StandardOpCode::OP_IMM, rd, rs, 1); }
    void cpop(Reg rd, Reg rs) { opShift(0x30, 1, StandardOpCode::OP_IMM, rd, rs, 2); }
    void sextb(Reg rd, Reg rs) { opShift(0x30, 1, StandardOpCode::OP_IMM, rd, rs, 4); }
    void sexth(Reg rd, Reg rs) { opShift(0x30, 1, StandardOpCode::OP_IMM, rd, rs, 5); }

    // FeNN vector processor

    // VSOP
    void vadd(VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b000, 0b0000000, rd, rs1, rs2); }
    void vadd_s(VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b000, 0b1000000, rd, rs1, rs2); }
    void vsub(VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b010, 0b0000000, rd, rs1, rs2); }
    void vsub_s(VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b010, 0b1000000, rd, rs1, rs2); }
    void vmul(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift, rd, rs1, rs2); }
    void vmul_rn(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift | 0b0010000, rd, rs1, rs2); }
    void vmul_rs(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift | 0b0100000, rd, rs1, rs2); }
    void vmul_s(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift | 0b1000000, rd, rs1, rs2); }
    void vmul_s_rn(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift | 0b1010000, rd, rs1, rs2); }
    void vmul_s_rs(Bit<4> shift, VReg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VSOP, 0b100, shift | 0b1100000, rd, rs1, rs2); }


    // VTST
    void vteq(Reg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VTST, 0b000, 0x0, rd, rs1, rs2); }
    void vtne(Reg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VTST, 0b010, 0x0, rd, rs1, rs2); }
    void vtlt(Reg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VTST, 0b100, 0x0, rd, rs1, rs2); }
    void vtge(Reg rd, VReg rs1, VReg rs2){ Rtype(VectorOpCode::VTST, 0b110, 0x0, rd, rs1, rs2); }

    // VSEL
    void vsel(VReg rd, const Reg &rs1, VReg rs2){ Rtype(VectorOpCode::VSEL, 0x0, 0x0, rd, rs1, rs2); }

    // VLUI
    void vlui(VReg rd, uint32_t imm){ Utype(VectorOpCode::VLUI, rd, imm); }

    // VMOV
    void vfill(VReg rd, Reg rs){ Itype(VectorOpCode::VMOV, 0b000, rd, rs, 0); }
    void vextract(Reg rd, VReg rs, Bit<5> lane){ Itype(VectorOpCode::VMOV, 0b001, rd, rs, lane); }

    // VSPC
    void vrng(VReg rd){ Itype(VectorOpCode::VSPC, 0x0, rd, Reg::X0, 0); }

    // VLOAD
    void vloadv(VReg rd, Reg addr, int imm = 0){ Itype(VectorOpCode::VLOAD, 0b000, rd, addr, imm); }
    void vloadr0(Reg addr, int imm = 0){ Itype(VectorOpCode::VLOAD, 0b001, 0, addr, imm); }
    void vloadr1(Reg addr, int imm = 0){ Itype(VectorOpCode::VLOAD, 0b101, 0, addr, imm); }
    
    // VSTORE
    void vstore(VReg rs, Reg addr, int imm = 0){ Stype(VectorOpCode::VSTORE, 0x0, addr, rs, imm); }

private:
    friend Label;

    //------------------------------------------------------------------------
    // ClabelVal
    //------------------------------------------------------------------------
    // for Label class
    struct ClabelVal {
        ClabelVal(uint32_t addr = 0) : addr(addr), refCount(1) {}
        const uint32_t addr;
        int refCount;
    };

    
    //----------------------------------------------------------------------------
    // Jmp
    //----------------------------------------------------------------------------
    struct Jmp 
    {
        enum Type {
            tJal,
            tBtype,
            tRawAddress,
        } type;
        const uint32_t from; /* address of the jmp mnemonic */
        uint32_t encoded;

        // jal
        Jmp(uint32_t from, Bit<7> opcode, Reg rd)
            : type(tJal)
            , from(from)
            , encoded((static_cast<uint32_t>(rd) << 7) | opcode)
        {
        }
        // B-type
        Jmp(uint32_t from, Bit<7> opcode, uint32_t funct3, Reg src1, Reg src2)
            : type(tBtype)
            , from(from)
            , encoded((static_cast<uint32_t>(src2) << 20) | (static_cast<uint32_t>(src1) << 15) | (funct3 << 12) | opcode)
        {
        }
        // raw address
        explicit Jmp(uint32_t from)
            : type(tRawAddress)
            , from(from)
            , encoded(0)
        {
        }

        uint32_t encode(uint32_t addr) const;

        // update jmp address by base->getCurr()
        void update(CodeGenerator *base) const;

        // append jmp opcode with addr
        void appendCode(CodeGenerator *base, uint32_t addr) const;
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::unordered_map<int, ClabelVal> ClabelDefList;
    typedef std::unordered_multimap<int, Jmp> ClabelUndefList;
    typedef std::unordered_set<Label*> LabelPtrList;

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void reset()
    {
        labelId_ = 1;
        clabelDefList_.clear();
        clabelUndefList_.clear();
        resetLabelPtrList();
    }
    void defineClabel(Label& label)
    {
        define_inner(clabelDefList_, clabelUndefList_, getId(label), getCurr());
        label.cg = this;
        labelPtrList_.insert(&label);
    }
    void assign(Label& dst, const Label& src)
    {
        ClabelDefList::const_iterator i = clabelDefList_.find(src.id);
        if (i == clabelDefList_.end()) throw Error(ERR_LABEL_IS_NOT_SET_BY_L);
        define_inner(clabelDefList_, clabelUndefList_, dst.id, i->second.addr);
        dst.cg = this;
        labelPtrList_.insert(&dst);
    }
    // return 0 unless label exists
    uint32_t getAddr(const Label& label) const
    {
        ClabelDefList::const_iterator i = clabelDefList_.find(getId(label));
        if (i == clabelDefList_.end()) return 0;
        return i->second.addr;
    }
    void addUndefinedLabel(const Label& label, const Jmp& jmp)
    {
        clabelUndefList_.insert(ClabelUndefList::value_type(label.id, jmp));
    }
    bool hasUndefClabel() const { return hasUndefinedLabel_inner(clabelUndefList_); }
    
    int getId(const Label& label) const
    {
        if (label.id == 0) label.id = labelId_++;
        return label.id;
    }
    void define_inner(ClabelDefList& defList, ClabelUndefList& undefList, int labelId, uint32_t addr)
    {
        // add label
        ClabelDefList::value_type item(labelId, addr);
        std::pair<ClabelDefList::iterator, bool> ret = defList.insert(item);
        if (!ret.second) throw Error(ERR_LABEL_IS_REDEFINED);
        // search undefined label
        for (;;) {
            ClabelUndefList::iterator itr = undefList.find(labelId);
            if (itr == undefList.end()) break;
            const Jmp& jmp = itr->second;
            jmp.update(this);
            undefList.erase(itr);
        }
    }

    void incRefCount(int id, Label *label)
    {
        clabelDefList_[id].refCount++;
        labelPtrList_.insert(label);
    }
    void decRefCount(int id, Label *label)
    {
        labelPtrList_.erase(label);
        ClabelDefList::iterator i = clabelDefList_.find(id);
        if (i == clabelDefList_.end()) return;
        if (i->second.refCount == 1) {
            clabelDefList_.erase(id);
        } else {
            --i->second.refCount;
        }
    }

    template<class T>
    bool hasUndefinedLabel_inner(const T& list) const
    {
        return !list.empty();
    }
    // detach all labels linked to LabelManager
    void resetLabelPtrList()
    {
        for (LabelPtrList::iterator i = labelPtrList_.begin(), ie = labelPtrList_.end(); i != ie; ++i) {
            (*i)->clear();
        }
        labelPtrList_.clear();
    }

    void append4B(uint32_t code) { code_.push_back(code); }
    
    void write4B(size_t offset, uint32_t v) 
    {
        assert((offset & 3) == 0);
        code_.at(offset / 4) = v; 
    }

    // **TODO**  add code base address
    uint32_t getCurr() const{ return static_cast<uint32_t>(code_.size()) * 4; }
    
    void opJmp(const Label& label, const Jmp& jmp)
    {
        const uint32_t addr = getAddr(label);
        jmp.appendCode(this, addr);
        if (addr) {
            return;
        }
        addUndefinedLabel(label, jmp);
    }
    uint32_t enc2(uint32_t a, uint32_t b) const { return (a<<7) | (b<<15); }
    uint32_t enc3(uint32_t a, uint32_t b, uint32_t c) const { return enc2(a, b) | (c<<20); }

    void Rtype(Bit<7> opcode, Bit<3> funct3, Bit<7> funct7, Bit<5> rd, Bit<5> rs1, Bit<5> rs2)
    {
        uint32_t v = (funct7<<25) | (funct3<<12) | opcode | enc3(rd, rs1, rs2);
        append4B(v);
    }
    void Itype(Bit<7> opcode, Bit<3> funct3, Bit<5> rd, Bit<5> rs1, int imm)
    {
        if (!inSBit(imm, 12)) {
            throw Error(ERR_IMM_IS_TOO_BIG);
        }
        uint32_t v = (imm<<20) | (funct3<<12) | opcode | enc2(rd, rs1);
        append4B(v);
    }
    void Stype(Bit<7> opcode, Bit<3> funct3, Bit<5> rs1, Bit<5> rs2, int imm)
    {
        if (!inSBit(imm, 12)) {
            throw Error(ERR_IMM_IS_TOO_BIG);
        }
        uint32_t v = ((imm>>5)<<25) | (funct3<<12) | opcode | enc3(imm & mask(5), rs1, rs2);
        append4B(v);
    }
    void Utype(Bit<7> opcode, Bit<5> rd, uint32_t imm)
    {
        if (imm >= (1u << 20)) {
            throw Error(ERR_IMM_IS_TOO_BIG);
        }
        uint32_t v = (imm<<12) | opcode | (rd<<7);
        append4B(v);
    }
    void opShift(Bit<7> pre, Bit<3> funct3, Bit<7> opcode, Bit<5> rd, Bit<5> rs1, uint32_t shamt, int range = 0)
    {
        if (range == 0) {
            range = 5;
        }
        if (shamt >= (1u << range)) {
            throw Error(ERR_IMM_IS_TOO_BIG);
        }
        uint32_t v = (pre<<25) | (funct3<<12) | opcode | enc3(rd, rs1, shamt);
        append4B(v);
    }
    
    void opCSR(Bit<32> baseValue, Bit<12> csr, Bit<5> rs1_uimm, Bit<5> rd)
    {
        /*
            31 .. 20 | 19 .. 15 | 14 .. 12 | 11 .. 7 | 6 .. 0
               csr     rs1_uimm     func3       rd     opcode

            func3 and opcode must be encoded in the baseValue
        */
        uint32_t v = (csr<<20) | (rs1_uimm<<15) | (rd<<7);
        v |= baseValue; // force-encode base value
        append4B(v);
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<uint32_t> code_;    
    mutable int labelId_;
    ClabelDefList clabelDefList_;
    ClabelUndefList clabelUndefList_;
    LabelPtrList labelPtrList_;
};

