#include "tokenizer.h"
#include <sstream>
#include <cctype>

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() {}

void Tokenizer::normalize_inplace(string& s) {
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') s[i] = (char)(c + 32);
        if (!((c >= '0' && c <= '9') || (s[i] >= 'a' && s[i] <= 'z') || s[i] == '-')) s[i] = ' ';
    }
    string r;
    bool prev_space = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char ch = s[i];
        if (ch == ' ') {
            if (prev_space) continue;
            prev_space = true;
            r.push_back(' ');
        } else {
            prev_space = false;
            r.push_back(ch);
        }
    }
    while (!r.empty() && r.front() == ' ') r.erase(r.begin());
    while (!r.empty() && r.back() == ' ') r.pop_back();
    s = move(r);
}

vector<string> Tokenizer::tokenize(const string& text) {
    string s = text;
    normalize_inplace(s);
    istringstream ss(s);
    vector<string> out;
    string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}
