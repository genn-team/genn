
#include <string>

//--------------------------------------------------------------------------
/*! \brief Function for converting code to contain only explicit single precision (float) constants 
 */
//--------------------------------------------------------------------------

string digits= string("0123456789");
string op= string("+-*/(<>= ,;")+string("\n")+string("\t");

void doFinal(string &code, unsigned int i, string type, unsigned int &state) 
{
    if (code[i] == 'f') {
	if (type == "double") {
	    code.erase(i,1);
	}
    }
    else {
	if (type == "float") {
	    if (i == code.size()-1) code= code+string("f");
	    else code.insert(i,1,'f');
	}
    }
    if (i < code.size()) {
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
	    if (digits.find(code[i]) == string::npos) {
		state= 0; // the number looks like an integer ...
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
	case 4: // we have had '.' and/or 'e', 'E', digits only now
	    if (digits.find(code[i]) == string::npos) {
		doFinal(code, i, type, state);
	    }
	    break;
	}
	i++;
//	if (i < code.size()) cerr << state << " " << code[i] << endl;
    }
    if (state > 1) {
	doFinal(code, i, type, state);
    }
//    cerr << endl;
//    cerr << code.size();
//    cerr << code << endl;
    return code;
}


