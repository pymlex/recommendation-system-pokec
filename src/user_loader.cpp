#include "user_loader.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

bool load_users_encoded(const string& users_encoded_csv,
                        const vector<string>& text_columns,
                        unordered_map<int, UserProfile>& out_profiles,
                        size_t max_users)
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
    while (getline(in, line) && c < 100000) {
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
        if (idx_age >= 0 && (size_t)idx_age < parts.size() && parts[idx_age].size())
            p.age = atoi(parts[idx_age].c_str());
        else p.age = 0;
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
    cout << "Loaded " << out_profiles.size() << " users total" << endl;
    return true;
}

int compute_median_age_from_profiles(const unordered_map<int, UserProfile>& profiles) {
    vector<int> ages;
    ages.reserve(profiles.size());
    for (auto &kv : profiles) {
        int a = kv.second.age;
        if (a > 0) ages.push_back(a);
    }
    if (ages.empty()) return 0;
    sort(ages.begin(), ages.end());
    size_t n = ages.size();
    if (n % 2) return ages[n/2];
    return (ages[n/2 - 1] + ages[n/2]) / 2;
}

bool load_median_age(const string& path, int& out_median) {
    out_median = 0;
    ifstream in(path);
    if (!in.is_open()) return false;
    string s;
    if (!getline(in, s)) { in.close(); return false; }
    out_median = atoi(s.c_str());
    in.close();
    return true;
}

bool save_median_age(const string& path, int median) {
    ofstream out(path);
    if (!out.is_open()) return false;
    out << median << "\n";
    out.close();
    return true;
}

int fill_missing_ages(unordered_map<int, UserProfile>& profiles, int median_age) {
    int cnt = 0;
    for (auto &kv : profiles) {
        if (kv.second.age == 0) {
            kv.second.age = median_age;
            ++cnt;
        }
    }
    return cnt;
}
