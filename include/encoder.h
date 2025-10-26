#ifndef ENCODER_H
#define ENCODER_H

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

struct Tokenizer;
struct Lemmatiser;

struct Encoder {
    Encoder(
        const vector<string>& colKeys,
        const unordered_map<string, unordered_map<string,int>>& token2id_per_col,
        const unordered_map<string,int>& club_to_id,
        const unordered_map<string,int>& address_part1_to_id,
        const unordered_map<string,int>& address_part2_to_id,
        const unordered_map<string,int>& address_part3_to_id,
        const unordered_map<int, vector<int>>& adjacency_in);

    void pass2(const string& profiles_tsv, const string& out_users_csv);

private:
    vector<string> colKeys;
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col;
    const unordered_map<string,int>& club_to_id;
    const unordered_map<string,int>& address_part1_to_id;
    const unordered_map<string,int>& address_part2_to_id;
    const unordered_map<string,int>& address_part3_to_id;
    const unordered_map<int, vector<int>>& adjacency;

    vector<string> split_line_to_cols(const string& line) const;
    string build_region_parts_csv(const string& raw_region) const;
    unordered_map<int,int> extract_club_counts_from_line(const string& line) const;
    string format_counts_to_csv(const unordered_map<int,int>& counts) const;
    string format_token_counts_to_csv(const unordered_map<int,int>& counts) const;
    vector<string> process_profile_line(const vector<string>& cols, Tokenizer& tok, Lemmatiser& lem) const;
};

#endif
