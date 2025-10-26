#include "serializer.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>

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
            // handle doubled quotes inside quoted field
            if (in_quote && i + 1 < line.size() && line[i+1] == '"') {
                cur.push_back('"'); ++i; continue;
            }
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

    // parse header to find indices
    vector<string> headers = split_csv_line(header);
    int idx_user = -1;
    int idx_public = -1;
    int idx_completion = -1;
    int idx_gender = -1;
    int idx_region = -1;
    int idx_age = -1;
    int idx_clubs = -1;
    int idx_friends = -1;

    vector<int> idx_token_cols; idx_token_cols.resize(num_token_cols, -1);

    for (size_t i = 0; i < headers.size(); ++i) {
        string h = headers[i];
        string hl = h;
        for (char &c : hl) if (c >= 'A' && c <= 'Z') c = char(c + ('a' - 'A'));
        if (hl == "user_id" || hl == "userid") idx_user = (int)i;
        else if (hl == "public" || hl == "ispublic" || hl == "public_flag") idx_public = (int)i;
        else if (hl == "completion_percentage" || hl == "completion") idx_completion = (int)i;
        else if (hl == "gender") idx_gender = (int)i;
        else if (hl == "region" || hl == "region_id") idx_region = (int)i;
        else if (hl == "age") idx_age = (int)i;
        else if (hl == "clubs") idx_clubs = (int)i;
        else if (hl == "friends") idx_friends = (int)i;
    }

    // try to find token columns by suffix "_tokens"
    for (size_t t = 0; t < headers.size(); ++t) {
        string hl = headers[t];
        for (char &c : hl) if (c >= 'A' && c <= 'Z') c = char(c + ('a' - 'A'));
        if (hl.size() > 7 && hl.substr(hl.size()-7) == "_tokens") {
            for (int j = 0; j < num_token_cols; ++j) {
                if (idx_token_cols[j] == -1) { idx_token_cols[j] = (int)t; break; }
            }
        }
    }

    // fallback positioning if tokens not found
    if (idx_token_cols[0] == -1) {
        int start = 0;
        vector<int> candidates = {idx_user, idx_public, idx_gender, idx_region, idx_age, idx_clubs, idx_friends};
        for (int v : candidates) if (v > start) start = v;
        int pos = start + 1;
        for (int j = 0; j < num_token_cols; ++j) {
            if (pos < (int)headers.size()) idx_token_cols[j] = pos++;
            else idx_token_cols[j] = -1;
        }
    }

    uint64_t offset = 0;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line(line);
        if (cols.size() == 0) continue;

        uint32_t user_id = 0;
        if (idx_user >= 0 && (size_t)idx_user < cols.size()) user_id = (uint32_t) atoi(cols[idx_user].c_str());

        uint32_t ispublic = 0;
        if (idx_public >= 0 && (size_t)idx_public < cols.size() && cols[idx_public].size()) ispublic = (uint32_t) atoi(cols[idx_public].c_str());

        uint32_t completion = 0;
        if (idx_completion >= 0 && (size_t)idx_completion < cols.size() && cols[idx_completion].size()) completion = (uint32_t) atoi(cols[idx_completion].c_str());

        uint32_t gender = 0;
        if (idx_gender >= 0 && (size_t)idx_gender < cols.size() && cols[idx_gender].size()) gender = (uint32_t) atoi(cols[idx_gender].c_str());

        // region: CSV has "p1;p2;p3" possibly quoted
        vector<uint32_t> region_parts;
        if (idx_region >= 0 && (size_t)idx_region < cols.size() && cols[idx_region].size()) {
            string rf = cols[idx_region];
            if (rf.size() >= 2 && rf.front() == '"' && rf.back() == '"') rf = rf.substr(1, rf.size()-2);
            stringstream rs(rf);
            string tok;
            while (getline(rs, tok, ';')) {
                if (tok.empty()) continue;
                region_parts.push_back((uint32_t) atoi(tok.c_str()));
            }
        }

        uint32_t age = 0;
        if (idx_age >= 0 && (size_t)idx_age < cols.size() && cols[idx_age].size()) {
            age = (uint32_t) atoi(cols[idx_age].c_str());
        }

        // clubs
        vector<uint32_t> clubs;
        if (idx_clubs >= 0 && (size_t)idx_clubs < cols.size() && cols[idx_clubs].size()) {
            string clubs_field = cols[idx_clubs];
            stringstream sc(clubs_field);
            string part;
            while (getline(sc, part, ';')) {
                if (part.size() == 0) continue;
                clubs.push_back((uint32_t) atoi(part.c_str()));
            }
        }

        // write record to binary
        uint32_t start_offset = (uint32_t) offset;

        // user_id
        bout.write(reinterpret_cast<const char*>(&user_id), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // ispublic
        bout.write(reinterpret_cast<const char*>(&ispublic), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // completion_percentage
        bout.write(reinterpret_cast<const char*>(&completion), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // gender
        bout.write(reinterpret_cast<const char*>(&gender), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // region: count + elements
        uint32_t region_count = (uint32_t) region_parts.size();
        bout.write(reinterpret_cast<const char*>(&region_count), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        for (uint32_t v : region_parts) {
            bout.write(reinterpret_cast<const char*>(&v), sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }

        // age as uint32_t
        bout.write(reinterpret_cast<const char*>(&age), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // clubs: count + elements
        uint32_t clubs_count = (uint32_t) clubs.size();
        bout.write(reinterpret_cast<const char*>(&clubs_count), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        for (size_t i = 0; i < clubs.size(); ++i) { uint32_t cid = clubs[i]; bout.write(reinterpret_cast<const char*>(&cid), sizeof(uint32_t)); offset += sizeof(uint32_t); }

        // token columns
        uint32_t cols_num = (uint32_t) num_token_cols;
        bout.write(reinterpret_cast<const char*>(&cols_num), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        for (int ci = 0; ci < num_token_cols; ++ci)
        {
            string field;
            if (idx_token_cols[ci] >= 0 && (size_t) idx_token_cols[ci] < cols.size()) field = cols[idx_token_cols[ci]];
            vector<pair<uint32_t,uint32_t>> pairs = parse_pairs(field);
            uint32_t pairs_count = (uint32_t) pairs.size();
            bout.write(reinterpret_cast<const char*>(&pairs_count), sizeof(uint32_t));
            offset += sizeof(uint32_t);
            for (size_t pi = 0; pi < pairs.size(); ++pi) {
                uint32_t tid = pairs[pi].first;
                uint32_t cnt = pairs[pi].second;
                bout.write(reinterpret_cast<const char*>(&tid), sizeof(uint32_t));
                offset += sizeof(uint32_t);
                bout.write(reinterpret_cast<const char*>(&cnt), sizeof(uint32_t));
                offset += sizeof(uint32_t);
            }
        }

        uint32_t rec_len = (uint32_t) (offset - start_offset);
        idxout << user_id << "," << start_offset << "," << rec_len << "\n";
    }

    bout.close();
    idxout.close();
    in.close();
    return true;
}
