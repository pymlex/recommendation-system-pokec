#include "lemmatizer_wrapper.h"
#include <cstdlib>
#include "../third_party/lemmagen/include/lemmagen.h"

Lemmatiser::Lemmatiser(const string& model_path) {
    int status = lem_load_language_library(model_path.c_str());
    if (status == STATUS_OK) loaded = true; else loaded = false;
}
Lemmatiser::~Lemmatiser() {
    if (loaded) lem_unload_language_library();
    loaded = false;
}
string Lemmatiser::lemmatize_word(const string& w) {
    if (!loaded) return string();
    char* out = lem_lemmatize_word_alloc(w.c_str());
    if (out == nullptr) return string();
    string res(out);
    free(out);
    return res;
}
vector<string> Lemmatiser::lemmatize_tokens(const vector<string>& toks) {
    vector<string> out;
    out.reserve(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        const string &w = toks[i];
        if (w.empty()) continue;
        string l = lemmatize_word(w);
        if (l.empty()) continue;
        out.push_back(l);
    }
    return out;
}
