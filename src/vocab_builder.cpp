#include "vocab_builder.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <algorithm>
#include <unordered_set>

using namespace std;
namespace fs = std::filesystem;

VocabBuilder::VocabBuilder(const vector<string> &colKeys_in) {
    colKeys = colKeys_in;
    for (const auto &key : colKeys) {
        token2id_per_col[key] = unordered_map<string,int>();
        docfreq_per_col[key] = unordered_map<int,int>();
    }
    club_to_id.clear();
    club_slug_to_title.clear();
    address_part1_to_id.clear();
    address_part2_to_id.clear();
    address_part3_to_id.clear();
}

string VocabBuilder::normalize_slug(const string& raw) {
    string slug;
    slug.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = (unsigned char)raw[i];
        if (c >= 'A' && c <= 'Z') slug.push_back((char)(c + ('a' - 'A')));
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || raw[i] == '-') slug.push_back(raw[i]);
        else if (!slug.empty() && slug.back() != '-') slug.push_back('-');
    }
    while (!slug.empty() && slug.back() == '-') slug.pop_back();
    return slug;
}

string VocabBuilder::normalize_address(const string& raw) {
    string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = (unsigned char)raw[i];
        if (c >= 'A' && c <= 'Z') out.push_back((char)(c + ('a' - 'A')));
        else out.push_back(raw[i]);
    }
    size_t a = 0;
    while (a < out.size() && isspace((unsigned char)out[a])) ++a;
    size_t b = out.size();
    while (b > a && isspace((unsigned char)out[b-1])) --b;
    return out.substr(a, b - a);
}

void VocabBuilder::process_line_clubs(const string& line) {
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>([^<]*)</a>");
    string::const_iterator start = line.cbegin();
    smatch m;
    while (regex_search(start, line.cend(), m, href_re)) {
        string raw_slug = m[1].str();
        string title = m[2].str();
        string slug = normalize_slug(raw_slug);
        if (slug.empty() && title.empty()) { start = m.suffix().first; continue; }
        if (club_to_id.find(slug) == club_to_id.end()) {
            int nid = (int)club_to_id.size();
            club_to_id[slug] = nid;
            club_slug_to_title[slug] = title;
        }
        start = m.suffix().first;
    }
}

void VocabBuilder::process_line_tokens(const vector<string>& cols, Tokenizer &tok, Lemmatiser &lem) {
    const size_t base_idx = 9;
    for (size_t ci = 0; ci < colKeys.size(); ++ci) {
        const string &key = colKeys[ci];
        size_t idx = base_idx + ci;
        if (idx >= cols.size()) continue;
        string text = cols[idx];
        if (text.empty() || text == "null") continue;
        vector<string> tokens = tok.tokenize(text);
        vector<string> lem_tokens = lem.lemmatize_tokens(tokens);
        unordered_set<int> seen_terms;
        for (const string &t : lem_tokens) {
            if (t.empty()) continue;
            auto it = token2id_per_col[key].find(t);
            if (it == token2id_per_col[key].end()) {
                int nid = (int)token2id_per_col[key].size();
                token2id_per_col[key][t] = nid;
                docfreq_per_col[key][nid] = 0;
                it = token2id_per_col[key].find(t);
            }
            int tid = it->second;
            if (seen_terms.find(tid) == seen_terms.end()) {
                docfreq_per_col[key][tid] += 1;
                seen_terms.insert(tid);
            }
        }
    }
}

void VocabBuilder::process_region_parts_from_cols(const vector<string>& cols) {
    if (cols.size() <= 4) return;
    string raw_region = cols[4];
    if (raw_region.empty() || raw_region == "null") return;
    string nr = normalize_address(raw_region);
    string part1, rest;
    size_t comma_pos = nr.find(',');
    if (comma_pos != string::npos) { part1 = nr.substr(0, comma_pos); rest = nr.substr(comma_pos + 1); }
    else { part1 = nr; rest.clear(); }
    auto trim = [](string &s){ size_t a = 0; while (a < s.size() && isspace((unsigned char)s[a])) ++a; size_t b = s.size(); while (b > a && isspace((unsigned char)s[b-1])) --b; s = s.substr(a, b - a); };
    trim(part1); trim(rest);
    string part2, part3;
    if (!rest.empty()) {
        size_t dash = rest.find('-');
        if (dash != string::npos) { part2 = rest.substr(0, dash); part3 = rest.substr(dash + 1); }
        else { part2 = rest; part3.clear(); }
    }
    trim(part2); trim(part3);
    if (!part1.empty() && part1 != "null") if (address_part1_to_id.find(part1) == address_part1_to_id.end()) address_part1_to_id[part1] = (int)address_part1_to_id.size();
    if (!part2.empty() && part2 != "null") if (address_part2_to_id.find(part2) == address_part2_to_id.end()) address_part2_to_id[part2] = (int)address_part2_to_id.size();
    if (!part3.empty() && part3 != "null") if (address_part3_to_id.find(part3) == address_part3_to_id.end()) address_part3_to_id[part3] = (int)address_part3_to_id.size();
}

static vector<string> split_csv_line_local(const string& line) {
    vector<string> out; string cur; bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') { if (in_quote && i + 1 < line.size() && line[i+1] == '"') { cur.push_back('"'); ++i; continue; } in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) { out.push_back(cur); cur.clear(); } else cur.push_back(c);
    }
    out.push_back(cur); return out;
}

bool VocabBuilder::load_vocab(const string &in_dir) {
    token2id_per_col.clear();
    docfreq_per_col.clear();
    club_to_id.clear();
    club_slug_to_title.clear();
    address_part1_to_id.clear();
    address_part2_to_id.clear();
    address_part3_to_id.clear();
    ifstream tok_in((fs::path(in_dir) / "tokens.csv").string());
    if (!tok_in.is_open()) return false;
    string header;
    if (!getline(tok_in, header)) return false;
    string line;
    while (getline(tok_in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line_local(line);
        if (cols.size() < 4) continue;
        string col = cols[0];
        string token = cols[1];
        int tid = atoi(cols[2].c_str());
        int df = atoi(cols[3].c_str());
        if (token2id_per_col.find(col) == token2id_per_col.end()) {
            token2id_per_col[col] = unordered_map<string,int>();
            docfreq_per_col[col] = unordered_map<int,int>();
        }
        token2id_per_col[col][token] = tid;
        docfreq_per_col[col][tid] = df;
    }
    tok_in.close();
    auto load_clubs = [&](const string &f) {
        ifstream in((fs::path(in_dir) / f).string());
        if (!in.is_open()) return;
        if (!getline(in, header)) { in.close(); return; }
        while (getline(in, line)) {
            if (line.empty()) continue;
            vector<string> cols = split_csv_line_local(line);
            if (cols.size() < 3) continue;
            int id = atoi(cols[0].c_str());
            string slug = cols[1];
            string title = cols[2];
            club_to_id[slug] = id;
            club_slug_to_title[slug] = title;
        }
        in.close();
    };
    load_clubs("clubs_map.csv");
    auto load_part = [&](const string &fname, unordered_map<string,int> &map) {
        ifstream in((fs::path(in_dir) / fname).string());
        if (!in.is_open()) return;
        if (!getline(in, header)) { in.close(); return; }
        while (getline(in, line)) {
            if (line.empty()) continue;
            vector<string> cols = split_csv_line_local(line);
            if (cols.size() < 2) continue;
            int id = atoi(cols[0].c_str());
            string val = cols[1];
            map[val] = id;
        }
        in.close();
    };
    load_part("addresses_part1.csv", address_part1_to_id);
    load_part("addresses_part2.csv", address_part2_to_id);
    load_part("addresses_part3.csv", address_part3_to_id);
    return true;
}

void VocabBuilder::pass1(const string &profiles_tsv, Tokenizer &tok, Lemmatiser &lem) {
    ifstream in(profiles_tsv);
    if (!in.is_open()) return;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> cols;
        string cell;
        stringstream ss(line);
        while (getline(ss, cell, '\t')) cols.push_back(cell);
        if (cols.empty()) continue;
        process_region_parts_from_cols(cols);
        process_line_clubs(line);
        process_line_tokens(cols, tok, lem);
    }
    in.close();
}

string VocabBuilder::csv_escape(const string& s) {
    bool need = false;
    for (char c : s) if (c == '"' || c == ',' || c == '\n' || c == '\r') { need = true; break; }
    if (!need) return s;
    string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\""; else out.push_back(c);
    }
    out += "\"";
    return out;
}

vector<string> VocabBuilder::split_csv_line(const string& line) {
    return split_csv_line_local(line);
}

void VocabBuilder::save_vocab(const string &out_dir) const {
    {
        ofstream tok_out(out_dir + "/tokens.csv");
        if (tok_out.is_open()) {
            tok_out << "column,token,tid,df\n";
            for (const auto &col : token2id_per_col) {
                const string &col_name = col.first;
                const auto &map_token2id = col.second;
                vector<pair<int,string>> inv;
                inv.reserve(map_token2id.size());
                for (const auto &p : map_token2id) inv.push_back(make_pair(p.second, p.first));
                sort(inv.begin(), inv.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
                for (const auto &pr : inv) {
                    int tid = pr.first;
                    const string &token = pr.second;
                    int df = 0;
                    auto dit = docfreq_per_col.find(col_name);
                    if (dit != docfreq_per_col.end()) {
                        auto dfit = dit->second.find(tid);
                        if (dfit != dit->second.end()) df = dfit->second;
                    }
                    bool need_quote = (token.find(',') != string::npos) || (token.find('"') != string::npos);
                    tok_out << col_name << ",";
                    if (need_quote) tok_out << '"';
                    for (size_t k = 0; k < token.size(); ++k) {
                        if (token[k] == '"') tok_out << "\"\""; else tok_out << token[k];
                    }
                    if (need_quote) tok_out << '"';
                    tok_out << "," << tid << "," << df << "\n";
                }
            }
            tok_out.close();
        }
    }
    {
        ofstream clubs_out(out_dir + "/clubs_map.csv");
        if (clubs_out.is_open()) {
            clubs_out << "club_id,slug,title\n";
            vector<pair<int,string>> id_slug;
            id_slug.reserve(club_to_id.size());
            for (auto it = club_to_id.begin(); it != club_to_id.end(); ++it) id_slug.push_back(make_pair(it->second, it->first));
            sort(id_slug.begin(), id_slug.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
            for (auto &pr : id_slug) {
                int id = pr.first; const string &slug = pr.second;
                const string &title = (club_slug_to_title.find(slug) != club_slug_to_title.end()) ? club_slug_to_title.at(slug) : string();
                bool need_quote = (title.find(',') != string::npos) || (title.find('"') != string::npos);
                clubs_out << id << "," << slug << ",";
                if (need_quote) clubs_out << '"';
                for (char ch : title) { if (ch == '"') clubs_out << "\"\""; else clubs_out << ch; }
                if (need_quote) clubs_out << '"';
                clubs_out << "\n";
            }
            clubs_out.close();
        }
    }
    auto write_part = [&](const unordered_map<string,int> &m, const string &fname, const string &hdr) {
        ofstream out((out_dir + "/" + fname).c_str());
        if (!out.is_open()) return;
        out << hdr << "\n";
        vector<pair<int,string>> v;
        v.reserve(m.size());
        for (auto &it : m) v.push_back(make_pair(it.second, it.first));
        sort(v.begin(), v.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
        for (auto &pr : v) {
            out << pr.first << ",";
            bool needq = (pr.second.find(',') != string::npos) || (pr.second.find('"') != string::npos);
            if (needq) out << '"';
            for (char ch : pr.second) { if (ch == '"') out << "\"\""; else out << ch; }
            if (needq) out << '"';
            out << "\n";
        }
    };
    write_part(address_part1_to_id, "addresses_part1.csv", "address_part1_id,address_part1");
    write_part(address_part2_to_id, "addresses_part2.csv", "address_part2_id,address_part2");
    write_part(address_part3_to_id, "addresses_part3.csv", "address_part3_id,address_part3");
}
