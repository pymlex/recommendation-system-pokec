#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <string>
#include <vector>
using namespace std;
struct Tokenizer {
    Tokenizer();
    ~Tokenizer();
    vector<string> tokenize(const string& text);
private:
    void normalize_inplace(string& s);
};
#endif
