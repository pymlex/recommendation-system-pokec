#ifndef ENCODER_H
#define ENCODER_H
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
struct Encoder {
    Encoder(const vector<string>& colKeys,
            const unordered_map<string, unordered_map<string,int>>& token2id_per_col,
            const unordered_map<string,int>& club_to_id);
    void pass2(const string& profiles_tsv, const string& out_users_csv);
private:
    vector<string> colKeys;
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col;
    const unordered_map<string,int>& club_to_id;
};
#endif
