#include "serializer.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <iostream>
using namespace std;

static vector<string> split_csv_line(const string& line)
{
    vector<string> out;
    string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        char c = line[i];
        if (c == '"' ) {
            in_quote = ! in_quote;
            continue;
        }
        if (c == ',' && ! in_quote) {
            out.push_back(cur);
            cur.clear();
        } else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static vector<pair<uint32_t,uint32_t>> parse_pairs(const string& s)
{
    vector<pair<uint32_t,uint32_t>> out;
    if (s.empty()) return out;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ';')) {
        if (token.empty()) continue;
        size_t pos = token.find(':');
        if (pos == string::npos) {
            uint32_t id = (uint32_t) atoi(token.c_str());
            out.push_back(make_pair(id, 1u));
        } else {
            uint32_t id = (uint32_t) atoi(token.substr(0,pos).c_str());
            uint32_t cnt = (uint32_t) atoi(token.substr(pos+1).c_str());
            out.push_back(make_pair(id, cnt));
        }
    }
    return out;
}

bool csv_to_bin_index(const string& users_csv, const string& out_bin, const string& out_index, int num_token_cols)
{
    ifstream in(users_csv);
    if (! in.is_open()) return false;
    ofstream bout(out_bin, ios::binary);
    ofstream idxout(out_index);
    if (! bout.is_open() || ! idxout.is_open()) return false;
    string header;
    if (! getline(in, header)) return false;
    uint64_t offset = 0;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line(line);
        if (cols.size() < 5) continue;
        uint32_t user_id = (uint32_t) atoi(cols[0].c_str());
        string gender = cols[1];
        string region = cols[2];
        string age = cols[3];
        string clubs_field = cols[4];
        vector<uint32_t> clubs;
        if (! clubs_field.empty()) {
            stringstream sc(clubs_field);
            string part;
            while (getline(sc, part, ';')) {
                if (part.size() == 0) continue;
                clubs.push_back((uint32_t) atoi(part.c_str()));
            }
        }
        uint32_t start_offset = (uint32_t) offset;
        // write user_id
        bout.write(reinterpret_cast<const char*>(&user_id), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        // write gender length + bytes
        uint32_t glen = (uint32_t) gender.size();
        uint32_t gllen = glen;
        bout.write(reinterpret_cast<const char*>(&gllen), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        if (glen) { bout.write(gender.data(), (streamsize) glen); offset += glen; }
        // write region length + bytes
        uint32_t rlen = (uint32_t) region.size();
        bout.write(reinterpret_cast<const char*>(&rlen), sizeof(uint32_t)); offset += sizeof(uint32_t);
        if (rlen) { bout.write(region.data(), (streamsize) rlen); offset += rlen; }
        // age
        uint32_t alen = (uint32_t) age.size();
        bout.write(reinterpret_cast<const char*>(&alen), sizeof(uint32_t)); offset += sizeof(uint32_t);
        if (alen) { bout.write(age.data(), (streamsize) alen); offset += alen; }
        // clubs
        uint32_t clubs_count = (uint32_t) clubs.size();
        bout.write(reinterpret_cast<const char*>(&clubs_count), sizeof(uint32_t)); offset += sizeof(uint32_t);
        for (size_t i = 0; i < clubs.size(); ++i) { uint32_t cid = clubs[i]; bout.write(reinterpret_cast<const char*>(&cid), sizeof(uint32_t)); offset += sizeof(uint32_t); }
        // token columns
        uint32_t cols_num = (uint32_t) num_token_cols;
        bout.write(reinterpret_cast<const char*>(&cols_num), sizeof(uint32_t)); offset += sizeof(uint32_t);
        for (int ci = 0; ci < num_token_cols; ++ci)
        {
            string field;
            if ((size_t) (5 + ci) < cols.size()) field = cols[5 + ci];
            vector<pair<uint32_t,uint32_t>> pairs = parse_pairs(field);
            uint32_t pairs_count = (uint32_t) pairs.size();
            bout.write(reinterpret_cast<const char*>(&pairs_count), sizeof(uint32_t)); offset += sizeof(uint32_t);
            for (size_t pi = 0; pi < pairs.size(); ++pi) {
                uint32_t tid = pairs[pi].first;
                uint32_t cnt = pairs[pi].second;
                bout.write(reinterpret_cast<const char*>(&tid), sizeof(uint32_t)); offset += sizeof(uint32_t);
                bout.write(reinterpret_cast<const char*>(&cnt), sizeof(uint32_t)); offset += sizeof(uint32_t);
            }
        }
        uint32_t rec_len = (uint32_t) (offset - start_offset);
        idxout << user_id << "," << start_offset << "," << rec_len << "\n";
    }
    return true;
}
