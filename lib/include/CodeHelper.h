
#ifndef __CODE_HELPER_CC
#define __CODE_HELPER_CC

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

#define SAVEP(X) "(" << X << ")" //! Macro for a "safe" output of a parameter into generated code by essentially just adding a bracket around the parameter value in the generated code.

#define OB(X) hlp.openBrace(X) //shortcut nomenclature to open the Xth curly brace { plus a new line
#define CB(X) hlp.closeBrace(X) //shortcut nomenclature to close the Xth curly brace } plus a new line
#define ENDL hlp.endl()//shortcut nomenclature to generate a newline followed correct number of indentation characters for the current level

class CodeHelper {
public:
    CodeHelper(): verbose(false) {
        braces.push_back(0);
    }

    void setVerbose(bool isVerbose) {
        verbose = isVerbose;
    }
    std::string openBrace(unsigned int level) {
        braces.push_back(level);
        if (verbose) printf("%sopen %u.\n",indentBy(braces.size() - 1).c_str(),level);
        std::string result  = " {\n";
        result.append(indentBy(braces.size() - 1));
        return  result;
    }

    std::string closeBrace(unsigned int level) {
        if (braces.back()==level) {
            if (verbose) printf("%sclose %u.\n",indentBy(braces.size() - 1).c_str(),level);
            braces.pop_back();
            std::string result  = "}\n";
            result.append(indentBy(braces.size() - 1));
            return result;
        } else {
            std::cerr << "Code generation error: Attempted to close brace " << level << ", expecting brace " << braces.back() << "\n" ;
            exit(1);
        }
    }

    std::string endl() const{
        std::string result =  "\n";
        //put out right number of tabs for level depth
        result.append(indentBy(braces.size() - 1));
        return result;
    }

private:
    std::string indentBy(unsigned int numIndents) const{
        std::string result =  ""; ///toString(numIndents);
        for (unsigned int i = 0; i < numIndents; i++) {
            result.append("    ");
        }
        return result;
    }

    std::vector<unsigned int>  braces;
    bool verbose;
};

extern CodeHelper hlp;

#endif
