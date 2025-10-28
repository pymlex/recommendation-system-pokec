#include "user_profile.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

using namespace std;

static vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool in_quote = false;
    out.reserve(128);
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') { in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static vector<pair<int,int>> parse_tok_field(const string& s) {
    vector<pair<int,int>> out;
    if (s.empty()) return out;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ';')) {
        if (token.empty()) continue;
        size_t pos = token.find(':');
        if (pos == string::npos) {
            int id = atoi(token.c_str());
            out.push_back(make_pair(id, 1));
        } else {
            int id = atoi(token.substr(0,pos).c_str());
            int cnt = atoi(token.substr(pos+1).c_str());
            out.push_back(make_pair(id, cnt));
        }
    }
    return out;
}

bool load_users_encoded(const string& users_encoded_csv,
                        const vector<string>& text_columns,
                        unordered_map<int, UserProfile>& out_profiles)
{
    out_profiles.clear();

    ifstream in(users_encoded_csv);
    if (! in.is_open()) return false;

    string header;
    if (! getline(in, header)) return false;

    int idx_user = 0;
    int idx_public = 1;
    int idx_completion = 2;
    int idx_gender = 3;
    int idx_region = 4;
    int idx_age = 5;
    int idx_clubs = 6;
    int idx_friends = 7;

    vector<int> idx_token_cols(text_columns.size(), -1);
    for (size_t t = 0; t < text_columns.size(); ++t) idx_token_cols[t] = (int)(8 + t);

    string line;
    int c = 0;
    while (getline(in, line) && c < 300000) {
        if (!(c % 10000)) {
            cout << "Loaded " << c << " users " << endl;
        }
        c++;

        if (line.empty()) continue;
        vector<string> parts = split_csv_line(line);
        if (parts.size() == 0) continue;

        int uid = atoi(parts[idx_user].c_str());
        if (uid == 0) continue;

        UserProfile p;
        p.user_id = uid;

        if (idx_public >= 0 && (size_t)idx_public < parts.size() && parts[idx_public].size())
            p.public_flag = atoi(parts[idx_public].c_str());
        else p.public_flag = -1;

        if (idx_completion >= 0 && (size_t)idx_completion < parts.size() && parts[idx_completion].size())
            p.completion_percentage = atoi(parts[idx_completion].c_str());
        else p.completion_percentage = -1;

        if (idx_gender >= 0 && (size_t)idx_gender < parts.size() && parts[idx_gender].size())
            p.gender = atoi(parts[idx_gender].c_str());
        else p.gender = -1;

        if (idx_age >= 0 && (size_t)idx_age < parts.size() && parts[idx_age].size()) {
            p.age = atoi(parts[idx_age].c_str());
        } else p.age = 0;

        if (idx_clubs >= 0 && (size_t)idx_clubs < parts.size() && parts[idx_clubs].size()) {
            stringstream sc(parts[idx_clubs]);
            string tok;
            while (getline(sc, tok, ';')) if (!tok.empty()) p.clubs.push_back((uint32_t)atoi(tok.c_str()));
        }

        if (idx_friends >= 0 && (size_t)idx_friends < parts.size() && parts[idx_friends].size()) {
            stringstream sf(parts[idx_friends]);
            string tok;
            while (getline(sf, tok, ';')) if (!tok.empty()) p.friends.push_back((uint32_t)atoi(tok.c_str()));
        }

        p.region_parts = { -1, -1, -1 };
        if (idx_region >= 0 && (size_t)idx_region < parts.size() && parts[idx_region].size()) {
            string rf = parts[idx_region];
            if (rf.size() >= 2 && rf.front() == '"' && rf.back() == '"') rf = rf.substr(1, rf.size()-2);
            stringstream rs(rf);
            string tok;
            int pi = 0;
            while (getline(rs, tok, ';') && pi < 3) {
                if (!tok.empty()) p.region_parts[pi] = atoi(tok.c_str());
                ++pi;
            }
            for (; pi < 3; ++pi) p.region_parts[pi] = -1;
        }

        p.token_cols.clear();
        p.token_cols.resize(text_columns.size());
        for (size_t t = 0; t < text_columns.size(); ++t) {
            int idx = idx_token_cols[t];
            if (idx >= 0 && (size_t)idx < parts.size() && parts[idx].size()) {
                vector<pair<int,int>> pairs = parse_tok_field(parts[idx]);
                for (auto &pr : pairs) p.token_cols[t][pr.first] = pr.second;
            }
        }

        out_profiles[p.user_id] = std::move(p);
    }

    in.close();
    return true;
}
