#ifndef LEMMATIZER_WRAPPER_H
#define LEMMATIZER_WRAPPER_H
#include <string>
#include <vector>
using namespace std;
struct Lemmatiser {
    Lemmatiser(const string& model_path);
    ~Lemmatiser();
    string lemmatize_word(const string& w);
    vector<string> lemmatize_tokens(const vector<string>& toks);
    bool loaded;
};
#endif
