#ifndef ENCODER_H
#define ENCODER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <regex>

using namespace std;

struct Tokenizer;
struct Lemmatiser;

struct Encoder {
public:
    Encoder(
        const vector<string>& colKeys,
        const unordered_map<string, unordered_map<string,int>>& token2id_per_col,
        const unordered_map<string,int>& club_to_id,
        const unordered_map<string,int>& address_to_id,
        const unordered_map<int, vector<int>>& adjacency_in);

    void pass2(const string& profiles_tsv, const string& out_users_csv);
    void save_addresses_map(const string& out_csv) const;

private:
    vector<string> colKeys;
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col;
    const unordered_map<string,int>& club_to_id;
    const unordered_map<string,int>& address_to_id;
    const unordered_map<int, vector<int>>& adjacency;

    unordered_map<string,int> region_to_id;

    vector<string> split_line_to_cols(const string& line) const;
    unordered_map<int,int> extract_club_counts_from_line(const string& line, const regex& href_re) const;
    string format_counts_to_csv(const unordered_map<int,int>& counts) const;
    string format_token_counts_to_csv(const unordered_map<int,int>& counts) const;
    unordered_map<int,int> encode_tokens_for_column(
        const string& text,
        const string& key,
        Tokenizer& tok,
        Lemmatiser& lem) const;
    int lookup_region_id(const string& region) const;
};
#endif
