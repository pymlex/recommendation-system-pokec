#ifndef VOCAB_BUILDER_H
#define VOCAB_BUILDER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"

using namespace std;

struct VocabBuilder {
    explicit VocabBuilder(const vector<string> &colKeys);

    void pass1(const string &profiles_tsv, Tokenizer &tok, Lemmatiser &lem);

    void save_vocab(const string &out_dir);

    unordered_map<string, unordered_map<string,int>> token2id_per_col;
    unordered_map<string, unordered_map<int,int>> docfreq_per_col;
    unordered_map<string,int> club_to_id;

private:
    vector<string> colKeys;
};

#endif
