#pragma once

// Standard C++ includes
#include <charconv>
#include <type_traits>

namespace MiniParse::Utils
{
    template<class... Ts> struct Overload : Ts... { using Ts::operator()...; };
    template<class... Ts> Overload(Ts...) -> Overload<Ts...>; // line not needed in

    template<typename T>
    T toCharsThrow(std::string_view input, int base = 10)
    {
        T out;
        std::from_chars_result result;
        if constexpr (std::is_floating_point_v<T>) {
            result = std::from_chars(input.data(), input.data() + input.size(), out,
                                     (base == 10) ? std::chars_format::general : std::chars_format::hex);
        }
        else {
            result = std::from_chars(input.data(), input.data() + input.size(), out, base);
        }
        
        if(result.ec == std::errc::invalid_argument) {
            throw std::invalid_argument("Unable to convert chars '" + std::string{input} + "'");
        }
        else if(result.ec == std::errc::result_out_of_range) {
            throw std::out_of_range("Unable to convert chars '" + std::string{input} + "'");
        }
        return out;
    }
}
