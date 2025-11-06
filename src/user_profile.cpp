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
