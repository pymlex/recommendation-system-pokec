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
    void save_vocab(const string &out_dir) const;
    bool load_vocab(const string &in_dir);

    unordered_map<string, unordered_map<string,int>> token2id_per_col;
    unordered_map<string, unordered_map<int,int>> docfreq_per_col;
    unordered_map<string,int> club_to_id;
    unordered_map<string,string> club_slug_to_title;
    unordered_map<string,int> address_part1_to_id;
    unordered_map<string,int> address_part2_to_id;
    unordered_map<string,int> address_part3_to_id;

private:
    vector<string> colKeys;
    void process_line_clubs(const string& line);
    void process_line_tokens(const vector<string>& cols, Tokenizer &tok, Lemmatiser &lem);
    void process_region_parts_from_cols(const vector<string>& cols);
    static string normalize_slug(const string& raw);
    static string normalize_address(const string& raw);
    static string csv_escape(const string& s);
    static vector<string> split_csv_line(const string& line);
};

#endif
