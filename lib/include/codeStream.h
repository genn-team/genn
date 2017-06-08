#pragma once

// Standard C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

// GeNN includes
#include "utils.h"

//#define OB(X) CodeStream::OB(X) //shortcut nomenclature to open the Xth curly brace { plus a new line
//#define CB(X) CodeStream::CB(X) //shortcut nomenclature to close the Xth curly brace } plus a new line
//#define ENDL std::endl  //shortcut nomenclature to generate a newline followed correct number of indentation characters for the current level

//----------------------------------------------------------------------------
// CodeStream
//----------------------------------------------------------------------------
// Code-generation helper which automatically inserts brackets, indents etc
// Based heavily on: https://stackoverflow.com/questions/15053753/writing-a-manipulator-for-a-custom-stream-class
class CodeStream : public std::ostream
{
private:
    //------------------------------------------------------------------------
    // IndentBuffer
    //------------------------------------------------------------------------
    class IndentBuffer : public std::streambuf
    {
    public:
        IndentBuffer(std::streambuf *sink) : m_Sink(sink), m_NewLine(false), m_IndentLevel(0){}

        //--------------------------------------------------------------------
        // Public API
        //--------------------------------------------------------------------
        void indent(){ m_IndentLevel++; }
        void deindent(){ m_IndentLevel--; }

    private:
        //--------------------------------------------------------------------
        // Streambuf overrides
        //--------------------------------------------------------------------
        virtual int_type overflow(int_type c) override
        {
            // If the character is an end-of-file, pass it directly to the sink
            if(traits_type::eq_int_type(c, traits_type::eof())) {
                return m_Sink->sputc(c);
            }

            // If last character was a newline, indent and clear flag
            if(m_NewLine) {
                std::fill_n(std::ostreambuf_iterator<char>(m_Sink), m_IndentLevel * 4, ' ');
                m_NewLine = false;
            }

            // If writing character to sink results in an end-of-file, return it
            if (traits_type::eq_int_type(m_Sink->sputc(c), traits_type::eof())) {
                return traits_type::eof();
            }

            // If character us a newline set flag
            if (traits_type::eq_int_type(c, traits_type::to_char_type('\n'))) {
                m_NewLine = true;
            }

            return traits_type::not_eof(c);
        }

        //----------------------------------------------------------------------------
        // Members
        //----------------------------------------------------------------------------
        std::streambuf *m_Sink;
        bool m_NewLine;
        unsigned int m_IndentLevel;
    };

public:
    //------------------------------------------------------------------------
    // OB
    //------------------------------------------------------------------------
    struct OB
    {
        OB(unsigned int level) : Level(level){}

        const unsigned int Level;
    };

    //------------------------------------------------------------------------
    // CB
    //------------------------------------------------------------------------
    struct CB
    {
        CB(unsigned int level) : Level(level){}

        const unsigned int Level;
    };

    CodeStream(std::ostream &stream): m_Buffer(stream.rdbuf()), std::ostream(&m_Buffer) {
        m_Braces.push_back(0);
    }

private:
    //------------------------------------------------------------------------
    // Friends
    //------------------------------------------------------------------------
    friend std::ostream&  operator << (std::ostream& s, const OB &ob);
    friend std::ostream&  operator << (std::ostream& s, const CB &cb);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<unsigned int> m_Braces;
    IndentBuffer m_Buffer;
};

//------------------------------------------------------------------------
// Operators
//------------------------------------------------------------------------
inline std::ostream& operator << (std::ostream& s, const CodeStream::OB &ob)
{
    CodeStream &c = dynamic_cast<CodeStream&>(s);

    // Write open bracket and endline to self
    c << "{" << std::endl;

    // Add brace to list
    c.m_Braces.push_back(ob.Level);

    // Increase indent
    c.m_Buffer.indent();

    return c;
}

inline std::ostream& operator << (std::ostream& s, const CodeStream::CB &cb)
{
    CodeStream &c = dynamic_cast<CodeStream&>(s);

    if (c.m_Braces.back() == cb.Level) {
        c.m_Braces.pop_back();

        // Decrease indent
        c.m_Buffer.deindent();

        // Write closed bracket and newline (setting flag)
        c << "}" << std::endl;
    } else {
        gennError("Code generation error: Attempted to close brace " + std::to_string(cb.Level) + ", expecting brace " + std::to_string(c.m_Braces.back()));
    }

    return c;
}