#include "encoder.h"
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>

using namespace std;

Encoder::Encoder(
    const vector<string>& colKeys_in,
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col_in,
    const unordered_map<string,int>& club_to_id_in,
    const unordered_map<string,int>& address_part1_to_id_in,
    const unordered_map<string,int>& address_part2_to_id_in,
    const unordered_map<string,int>& address_part3_to_id_in,
    const unordered_map<int, vector<int>>& adjacency_in)
    : colKeys(colKeys_in),
      token2id_per_col(token2id_per_col_in),
      club_to_id(club_to_id_in),
      address_part1_to_id(address_part1_to_id_in),
      address_part2_to_id(address_part2_to_id_in),
      address_part3_to_id(address_part3_to_id_in),
      adjacency(adjacency_in)
{}

vector<string> Encoder::split_line_to_cols(const string& line) const {
    vector<string> cols;
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, '\t')) cols.push_back(cell);
    return cols;
}

string Encoder::build_region_parts_csv(const string& raw_region) const {
    string nr = raw_region;
    for (size_t i = 0; i < nr.size(); ++i) {
        unsigned char c = (unsigned char)nr[i];
        if (c >= 'A' && c <= 'Z') nr[i] = (char)(c + ('a' - 'A'));
    }
    string part1, rest;
    size_t comma_pos = nr.find(',');
    if (comma_pos == string::npos) { part1 = nr; rest.clear(); } else { part1 = nr.substr(0, comma_pos); rest = nr.substr(comma_pos+1); }
    auto trim = [](string &s){ size_t a=0; while (a<s.size() && isspace((unsigned char)s[a])) ++a; size_t b=s.size(); while (b>a && isspace((unsigned char)s[b-1])) --b; s = s.substr(a,b-a); };
    trim(part1); trim(rest);
    string part2, part3;
    if (!rest.empty()) {
        size_t dash = rest.find('-');
        if (dash == string::npos) { part2 = rest; part3.clear(); } else { part2 = rest.substr(0,dash); part3 = rest.substr(dash+1); }
    }
    trim(part2); trim(part3);
    int p1v = -1, p2v = -1, p3v = -1;
    auto it1 = address_part1_to_id.find(part1); if (it1 != address_part1_to_id.end()) p1v = it1->second;
    auto it2 = address_part2_to_id.find(part2); if (it2 != address_part2_to_id.end()) p2v = it2->second;
    auto it3 = address_part3_to_id.find(part3); if (it3 != address_part3_to_id.end()) p3v = it3->second;
    string out;
    if (p1v >= 0) out += to_string(p1v);
    out += ";";
    if (p2v >= 0) out += to_string(p2v);
    out += ";";
    if (p3v >= 0) out += to_string(p3v);
    return out;
}

unordered_map<int,int> Encoder::extract_club_counts_from_line(const string& line) const {
    unordered_map<int,int> club_counts;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>");
    string::const_iterator start = line.cbegin();
    smatch m;
    while (regex_search(start, line.cend(), m, href_re)) {
        string raw = m[1].str();
        string slug; slug.reserve(raw.size());
        for (char ch : raw) {
            unsigned char c = (unsigned char)ch;
            if (c >= 'A' && c <= 'Z') slug.push_back((char)(c + ('a' - 'A')));
            else slug.push_back(ch);
        }
        auto it = club_to_id.find(slug);
        if (it != club_to_id.end()) club_counts[it->second] += 1;
        start = m.suffix().first;
    }
    return club_counts;
}

string Encoder::format_counts_to_csv(const unordered_map<int,int>& counts) const {
    string out; bool first = true;
    for (auto &p : counts) {
        if (!first) out.push_back(';'); first = false;
        out += to_string(p.first);
    }
    return out;
}

string Encoder::format_token_counts_to_csv(const unordered_map<int,int>& counts) const {
    if (counts.empty()) return string();
    string out; bool first = true;
    for (auto &p : counts) {
        if (!first) out.push_back(';'); first = false;
        out += to_string(p.first); out.push_back(':'); out += to_string(p.second);
    }
    return out;
}

vector<string> Encoder::process_profile_line(const vector<string>& cols, Tokenizer& tok, Lemmatiser& lem) const {
    vector<string> outrow;
    if (cols.empty()) return outrow;
    int uid = atoi(cols[0].c_str());
    string pub = cols.size()>1 ? cols[1] : "";
    string comp = cols.size()>2 ? cols[2] : "";
    string gender = cols.size()>3 ? cols[3] : "";
    string region_csv = cols.size()>4 ? build_region_parts_csv(cols[4]) : string(";;");
    string age = cols.size()>7 ? cols[7] : "0";
    string clubs = format_counts_to_csv(extract_club_counts_from_line(cols.empty() ? string() : cols[0] + "\t" + (cols.size()>1?cols[1]:"")));
    if (cols.size() > 0) clubs = format_counts_to_csv(extract_club_counts_from_line(cols.back()));
    string clubs2 = format_counts_to_csv(extract_club_counts_from_line(cols.empty() ? string() : cols.back()));
    string friends;
    auto it = adjacency.find(uid);
    if (it != adjacency.end()) {
        for (size_t i = 0; i < it->second.size(); ++i) {
            if (i) friends.push_back(';');
            friends += to_string(it->second[i]);
        }
    }
    vector<string> token_cols;
    token_cols.resize(colKeys.size());
    for (size_t i = 0; i < colKeys.size(); ++i) {
        size_t idx = 9 + i;
        string text = idx < cols.size() ? cols[idx] : "";
        if (text.empty() || text == "null") { token_cols[i] = string(); continue; }
        vector<string> toks = tok.tokenize(text);
        vector<string> lems = lem.lemmatize_tokens(toks);
        unordered_map<int,int> counts;
        auto itmap = token2id_per_col.find(colKeys[i]);
        if (itmap != token2id_per_col.end()) {
            for (auto &w : lems) {
                auto jt = itmap->second.find(w);
                if (jt != itmap->second.end()) counts[jt->second] += 1;
            }
        }
        token_cols[i] = format_token_counts_to_csv(counts);
    }
    outrow.push_back(to_string(uid));
    outrow.push_back(pub);
    outrow.push_back(comp);
    outrow.push_back(gender);
    outrow.push_back(region_csv);
    outrow.push_back(age);
    outrow.push_back(clubs2);
    outrow.push_back(friends);
    for (auto &tc : token_cols) outrow.push_back(tc);
    return outrow;
}

void Encoder::pass2(const string& profiles_tsv, const string& out_users_csv) {
    ifstream in(profiles_tsv);
    if (!in.is_open()) return;
    Tokenizer tok;
    Lemmatiser lem("data/lem-me-sk.bin");
    vector<vector<string>> rows;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        auto cols = split_line_to_cols(line);
        if (cols.empty()) continue;
        auto row = process_profile_line(cols, tok, lem);
        if (!row.empty()) rows.push_back(row);
    }
    in.close();
    ofstream out(out_users_csv);
    out << "user_id,public,completion_percentage,gender,region,age,clubs,friends";
    for (auto &k : colKeys) out << "," << k << "_tokens";
    out << "\n";
    for (auto &r : rows) {
        for (size_t i = 0; i < r.size(); ++i) {
            out << r[i];
            if (i+1 < r.size()) out << ",";
        }
        out << "\n";
    }
    out.close();
}
