#pragma once

// Standard C++ includes
#include <chrono>
#include <iostream>
#include <string>

//------------------------------------------------------------------------
// SpineMLSimulator::Timer
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::steady_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_Start).count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::steady_clock> m_Start;
    std::string m_Title;
};
}