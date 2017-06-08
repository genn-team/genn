#pragma once

// Standard C++ includes
#include <ostream>
#include <streambuf>
#include <string>
#include <vector>

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
        IndentBuffer() : m_Sink(NULL), m_NewLine(false), m_IndentLevel(0){}

        //--------------------------------------------------------------------
        // Public API
        //--------------------------------------------------------------------
        void indent()
        {
            m_IndentLevel++;
        }

        void deindent()
        {
            m_IndentLevel--;
        }

        void setSink(std::streambuf *sink)
        {
            m_Sink = sink;
        }

    private:
        //--------------------------------------------------------------------
        // Streambuf overrides
        //--------------------------------------------------------------------
        virtual int overflow(int c) override;

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

    CodeStream(std::ostream &stream): std::ostream(&m_Buffer) {
        m_Buffer.setSink(stream.rdbuf());
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
    IndentBuffer m_Buffer;
    std::vector<unsigned int> m_Braces;
};

//------------------------------------------------------------------------
// Operators
//------------------------------------------------------------------------
std::ostream& operator << (std::ostream& s, const CodeStream::OB &ob);
std::ostream& operator << (std::ostream& s, const CodeStream::CB &cb);