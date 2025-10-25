#include "user_profile.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace std;

static string lower_copy(const string& s) {
    string r = s;
    for (size_t i = 0; i < r.size(); ++i) {
        unsigned char c = (unsigned char) r[i];
        if (c >= 'A' && c <= 'Z') r[i] = (char)(c + ('a' - 'A'));
    }
    return r;
}

static vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool in_quote = false;
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
                        unordered_map<int, UserProfile>& out_profiles,
                        int& out_median_age)
{
    out_profiles.clear();
    out_median_age = 0;

    ifstream in(users_encoded_csv);
    if (! in.is_open()) return false;

    string header;
    if (! getline(in, header)) return false;

    vector<string> headers = split_csv_line(header);
    vector<string> headers_l;
    headers_l.reserve(headers.size());
    for (auto &h : headers) headers_l.push_back(lower_copy(h));

    int idx_user = -1, idx_public = -1, idx_gender = -1, idx_region = -1, idx_age = -1, idx_clubs = -1, idx_friends = -1;
    vector<int> idx_token_cols; idx_token_cols.resize(text_columns.size(), -1);

    for (size_t i = 0; i < headers_l.size(); ++i) {
        const string &h = headers_l[i];
        if (h == "user_id" || h == "userid") idx_user = (int)i;
        else if (h == "public" || h == "public_flag") idx_public = (int)i;
        else if (h == "gender") idx_gender = (int)i;
        else if (h == "region_id" || h == "region") idx_region = (int)i;
        else if (h == "age" || h == "age" || h == "a g e") idx_age = (int)i;
        else if (h == "clubs") idx_clubs = (int)i;
        else if (h == "friends") idx_friends = (int)i;
    }

    for (size_t t = 0; t < text_columns.size(); ++t) {
        string key = lower_copy(text_columns[t]) + "_tokens";
        for (size_t i = 0; i < headers_l.size(); ++i) if (headers_l[i] == key) { idx_token_cols[t] = (int)i; break; }
    }

    string line;
    vector<int> ages_nonzero;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> parts = split_csv_line(line);
        if (parts.size() == 0) continue;

        UserProfile p;
        if (idx_user >= 0 && (size_t)idx_user < parts.size() && parts[idx_user].size()) {
            p.user_id = atoi(parts[idx_user].c_str());
        } else continue; 

        if (idx_public >= 0 && (size_t)idx_public < parts.size() && parts[idx_public].size()) {
            p.public_flag = atoi(parts[idx_public].c_str());
        } else p.public_flag = -1;

        if (idx_gender >= 0 && (size_t)idx_gender < parts.size() && parts[idx_gender].size()) {
            string g = parts[idx_gender];
            // normalize small
            p.gender = g;
        } else p.gender.clear();

        if (idx_region >= 0 && (size_t)idx_region < parts.size() && parts[idx_region].size()) {
            p.region_id = atoi(parts[idx_region].c_str());
        } else p.region_id = -1;

        if (idx_age >= 0 && (size_t)idx_age < parts.size() && parts[idx_age].size()) {
            p.age = atoi(parts[idx_age].c_str());
        } else p.age = 0;
        if (p.age > 0) ages_nonzero.push_back(p.age);

        if (idx_clubs >= 0 && (size_t)idx_clubs < parts.size() && parts[idx_clubs].size()) {
            string clubs_field = parts[idx_clubs];
            stringstream sc(clubs_field);
            string tok;
            while (getline(sc, tok, ';')) {
                if (tok.size()) p.clubs.push_back(atoi(tok.c_str()));
            }
        }

        if (idx_friends >= 0 && (size_t)idx_friends < parts.size() && parts[idx_friends].size()) {
            string friends_field = parts[idx_friends];
            stringstream sf(friends_field);
            string tok;
            while (getline(sf, tok, ';')) {
                if (tok.size()) p.friends.push_back(atoi(tok.c_str()));
            }
        }

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

    if (ages_nonzero.empty()) out_median_age = 0;
    else {
        sort(ages_nonzero.begin(), ages_nonzero.end());
        size_t n = ages_nonzero.size();
        if (n % 2) out_median_age = ages_nonzero[n/2];
        else out_median_age = (ages_nonzero[n/2 - 1] + ages_nonzero[n/2]) / 2;
    }

    // replace zeros with median
    for (auto &kv : out_profiles) {
        if (kv.second.age == 0) kv.second.age = out_median_age;
    }

    return true;
}
