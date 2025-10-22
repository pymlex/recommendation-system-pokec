#include "lemmatizer_wrapper.h"
#include <cstring>
#include "../third_party/lemmagen/include/lemmagen.h"
using namespace std;

Lemmatiser::Lemmatiser(const string& model_path)
{
    int status = lem_load_language_library(model_path.c_str());
    if (status == STATUS_OK) loaded = true; else loaded = false;
}
Lemmatiser::~Lemmatiser()
{
    if (loaded) lem_unload_language_library();
    loaded = false;
}
string Lemmatiser::lemmatize_word(const string& w)
{
    if (! loaded) return string();
    char* out = lem_lemmatize_word_alloc(w.c_str());
    if (out == nullptr) return string();
    string res(out);
    free(out);
    return res;
}
vector<string> Lemmatiser::lemmatize_tokens(const vector<string>& toks)
{
    vector<string> out;
    for (size_t i = 0; i < toks.size(); ++i)
    {
        string w = toks[i];
        if (w.size() == 0) continue;
        string l = lemmatize_word(w);
        if (l.size() == 0) continue;
        out.push_back(l);
    }
    return out;
}
