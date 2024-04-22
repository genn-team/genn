#include "code_generator/codeStream.h"

// Standard C++ includes
#include <algorithm>

//------------------------------------------------------------------------
// GeNN::CodeGenerator::CodeStream::IndentBuffer
//------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
int CodeStream::IndentBuffer::overflow(int c)
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

//------------------------------------------------------------------------
// GeNN::CodeGenerator::CodeStream::Scope
//------------------------------------------------------------------------
unsigned int CodeStream::Scope::s_NextLevel = 0;

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
std::ostream& operator << (std::ostream& s, const CodeStream::OB &ob)
{
    CodeStream &c = dynamic_cast<CodeStream&>(s);

    // Write open bracket and endline to self
    c << " {" << std::endl;

    // Add brace to list
    c.m_Braces.push_back(ob.Level);

    // Increase indent
    c.m_Buffer.indent();

    return c;
}

std::ostream& operator << (std::ostream& s, const CodeStream::CB &cb)
{
    CodeStream &c = dynamic_cast<CodeStream&>(s);

    if (c.m_Braces.back() == cb.Level) {
        c.m_Braces.pop_back();

        // Decrease indent
        c.m_Buffer.deindent();

        // Write closed bracket and newline (setting flag)
        c << "}" << std::endl;
    } else {
        throw std::runtime_error("Code generation error: Attempted to close brace " + std::to_string(cb.Level) + ", expecting brace " + std::to_string(c.m_Braces.back()));
    }

    return c;
}
}
