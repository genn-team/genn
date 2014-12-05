
#include <string>

//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string trg, const string rep)
{
  size_t found= s.find(trg);
  while (found != string::npos) {
    s.replace(found,trg.length(),rep);
    found= s.find(trg);
  }
}


void name_substitutions(string &code, string prefix, vector<string> &names, string postfix= string(""))
{
    for (int k = 0, l = names.size(); k < l; k++) {
	substitute(code, tS("$(") + names[k] + tS(")"), prefix+names[k]+postfix);
    }
}

void value_substitutions(string &code, vector<string> &names, vector<double> &values)
{
    for (int k = 0, l = names.size(); k < l; k++) {
	substitute(code, tS("$(") + names[k] + tS(")"), tS(values[k]));
    }
}

void extended_name_substitutions(string &code, string prefix, vector<string> &names, string ext, string postfix= string(""))
{
    for (int k = 0, l = names.size(); k < l; k++) {
	substitute(code, tS("$(") + names[k] + ext + tS(")"), prefix+names[k]+postfix);
    }
}

void extended_value_substitutions(string &code, vector<string> &names, string ext, vector<double> &values)
{
    for (int k = 0, l = names.size(); k < l; k++) {
	substitute(code, tS("$(") + names[k] + ext + tS(")"), tS(values[k]));
    }
}

//--------------------------------------------------------------------------
/*! \brief Function for converting code to contain only explicit single precision (float) constants 
 */
//--------------------------------------------------------------------------

string digits= string("0123456789");
string op= string("+-*/(<>= ,;")+string("\n")+string("\t");

#define __mathFN 56
const char *__dnames[__mathFN]= {
  "cos",
  "sin",
  "tan",
  "acos",
  "asin",
  "atan",
  "atan2",
  "cosh",
  "sinh",
  "tanh",
  "acosh",
  "asinh",
  "atanh",
  "exp",
  "frexp",
  "ldexp",
  "log",
  "log10",
  "modf",
  "exp2",
  "expm1",
  "ilogb",
  "log1p",
  "log2",
  "logb",
  "scalbn",
  "scalbln",
  "pow",
  "sqrt",
  "cbrt",
  "hypot",
  "erf",
  "erfc",
  "tgamma",
  "lgamma",
  "ceil",
  "floor",
  "fmod",
  "trunc",
  "round",
  "lround",
  "llround",
  "rint",
  "lrint",
  "nearbyint",
  "remainder",
  "remquo",
  "copysign",
  "nan",
  "nextafter",
  "nexttoward",
  "fdim",
  "fmax",
  "fmin",
  "fabs",
  "fma"
};

const char *__fnames[__mathFN]= {
  "cosf",
  "sinf",
  "tanf",
  "acosf",
  "asinf",
  "atanf",
  "atan2f",
  "coshf",
  "sinhf",
  "tanhf",
  "acoshf",
  "asinhf",
  "atanhf",
  "expf",
  "frexpf",
  "ldexpf",
  "logf",
  "log10f",
  "modff",
  "exp2f",
  "expm1f",
  "ilogbf",
  "log1pf",
  "log2f",
  "logbf",
  "scalbnf",
  "scalblnf",
  "powf",
  "sqrtf",
  "cbrtf",
  "hypotf",
  "erff",
  "erfcf",
  "tgammaf",
  "lgammaf",
  "ceilf",
  "floorf",
  "fmodf",
  "truncf",
  "roundf",
  "lroundf",
  "llroundf",
  "rintf",
  "lrintf",
  "nearbyintf",
  "remainderf",
  "remquof",
  "copysignf",
  "nanf",
  "nextafterf",
  "nexttowardf",
  "fdimf",
  "fmaxf",
  "fminf",
  "fabsf",
  "fmaf"
};

void ensureMathFunctionFtype(string &code, string type) 
{
  if (type == string("double")) {
    for (int i= 0; i < __mathFN; i++) {
      substitute(code, string(__fnames[i])+string("("), string(__dnames[i])+string("(")); 
    }
  }
  else {
    for (int i= 0; i < __mathFN; i++) {
      substitute(code, string(__dnames[i])+string("("), string(__fnames[i])+string("(")); 
    }
  }
}


void doFinal(string &code, unsigned int i, string type, unsigned int &state) 
{
    if (code[i] == 'f') {
	if (type == "double") {
	    code.erase(i,1);
	}
    }
    else {
	if (type == "float") {
	    code.insert(i,1,'f');
	}
    }
    if (i < code.size()-1) {
	if (op.find(code[i]) == string::npos) {
	    state= 0;
	}
	else {
	    state= 1;
	}
    }
}


string ensureFtype(string oldcode, string type) 
{
//    cerr << "entering ensure" << endl;
//    cerr << oldcode << endl;
    string code= oldcode;
    unsigned int i= 0;
    unsigned int state= 1; // allowed to start with a number straight away.
    while (i < code.size()) {
	switch (state) 
	{
	case 0: // looking for a valid lead-in
	    if (op.find(code[i]) != string::npos) {
		state= 1;
		break;
	    }
	    break;
	case 1: // looking for start of number
	    if (digits.find(code[i]) != string::npos) {
		state= 2; // found the beginning of a number starting with a digit
		break;
	    }
	    if (code[i] == '.') {
		state= 3; // number starting with a dot
		break;
	    }
	    if (op.find(code[i]) == string::npos) {
		state= 0;
		break;
	    }
	    break;
	case 2: // in a number, looking for more digits, '.', 'e', 'E', or end of number
	    if (code[i] == '.') {
		state= 3; // number now also contained a dot
		break;
	    }
	    if ((code[i] == 'e') || (code[i] == 'E')) {
		state= 4;
		break;
	    }
	    if (digits.find(code[i]) == string::npos) {// the number looks like an integer ...
		if (op.find(code[i]) != string::npos) state= 1;
		else state= 0; 
		break;
	    }
	    break;
	case 3: // we have had '.' now looking for digits or 'e', 'E'
	    if ((code[i] == 'e') || (code[i] == 'E')) {
		state= 4;
		break;
	    }
	    if (digits.find(code[i]) == string::npos) {
		doFinal(code, i, type, state);
		break;
	    }
	    break;
	case 4: // we have had '.' and 'e', 'E', digits only now
	    if (digits.find(code[i]) != string::npos) {
		state= 6;
		break;
	    }
	    if ((code[i] != '+') && (code[i] != '-')) {
		if (op.find(code[i]) != string::npos) state= 1;
		else state= 0; 
		break;
	    }
	    else {
		state= 5;
		break;
	    }
	case 5: // now one or more digits or else ...
	    if (digits.find(code[i]) != string::npos) {
		state= 6;
		break;
	    }
	    else {
		if (op.find(code[i]) != string::npos) state= 1;
		else state= 0; 
		break;
	    }
	case 6: // any non-digit character will trigger action
	    if (digits.find(code[i]) == string::npos) {
		doFinal(code, i, type, state);
		break;
	    }
	    break;
	}
	i++;
    }
    if ((state == 3) || (state == 6)) {
	if (type == "float") {
	    code= code+string("f");
	}
    }
    ensureMathFunctionFtype(code, type);
    return code;
}


