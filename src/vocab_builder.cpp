#include "vocab_builder.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <algorithm>
#include <unordered_set>

using namespace std;
namespace fs = std::filesystem;

VocabBuilder::VocabBuilder(const vector<string> &colKeys_in)
{
    colKeys = colKeys_in;
    for (const auto &key : colKeys) {
        token2id_per_col[key] = unordered_map<string,int>();
        docfreq_per_col[key] = unordered_map<int,int>();
    }
    club_to_id.clear();
    club_slug_to_title.clear();
    address_to_id.clear();
}

string VocabBuilder::normalize_slug(const string& raw)
{
    string slug;
    slug.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = (unsigned char) raw[i];
        if (c >= 'A' && c <= 'Z') slug.push_back((char)(c + ('a' - 'A')));
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || raw[i] == '-') slug.push_back(raw[i]);
        else if (!slug.empty() && slug.back() != '-') slug.push_back('-');
    }
    while (!slug.empty() && slug.back() == '-') slug.pop_back();
    return slug;
}

string VocabBuilder::normalize_address(const string& raw)
{
    string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = (unsigned char) raw[i];
        if (c >= 'A' && c <= 'Z') out.push_back((char)(c + ('a' - 'A')));
        else out.push_back(raw[i]);
    }
    size_t a = 0;
    while (a < out.size() && isspace((unsigned char)out[a])) ++a;
    size_t b = out.size();
    while (b > a && isspace((unsigned char)out[b-1])) --b;
    return out.substr(a, b - a);
}

void VocabBuilder::process_line_clubs(const string& line)
{
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>([^<]*)</a>");
    string::const_iterator start = line.cbegin();
    smatch m;
    while (regex_search(start, line.cend(), m, href_re)) {
        string raw_slug = m[1].str();
        string title = m[2].str();
        string slug = normalize_slug(raw_slug);
        if (slug.empty() && title.empty()) {
            start = m.suffix().first;
            continue;
        }
        if (club_to_id.find(slug) == club_to_id.end()) {
            int nid = (int) club_to_id.size();
            club_to_id[slug] = nid;
            club_slug_to_title[slug] = title;
        }
        start = m.suffix().first;
    }
}

void VocabBuilder::process_line_tokens(const vector<string>& cols, Tokenizer &tok, Lemmatiser &lem)
{
    const size_t base_idx = 9; // I_am_working_in_field index (0-based)
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

void VocabBuilder::pass1(const string &profiles_tsv, Tokenizer &tok, Lemmatiser &lem)
{
    ifstream in(profiles_tsv);
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> cols;
        string cell;
        stringstream ss(line);
        while (getline(ss, cell, '\t')) cols.push_back(cell);
        if (cols.size() == 0) continue;
        if (cols.size() > 4) {
            string raw_region = cols[4];
            if (!raw_region.empty() && raw_region != "null") {
                string nr = normalize_address(raw_region);
                if (!nr.empty() && address_to_id.find(nr) == address_to_id.end()) {
                    int nid = (int) address_to_id.size();
                    address_to_id[nr] = nid;
                }
            }
        }
        process_line_clubs(line);
        process_line_tokens(cols, tok, lem);
    }
    in.close();
}

// Utilities
string VocabBuilder::csv_escape(const string& s) {
    bool need = false;
    for (char c : s) if (c == '"' || c == ',' || c == '\n' || c == '\r') { need = true; break; }
    if (!need) return s;
    string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

// split CSV line taking quotes into account
vector<string> VocabBuilder::split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            // handle doubled quotes
            if (in_quote && i + 1 < line.size() && line[i+1] == '"') {
                cur.push_back('"'); ++i; continue;
            }
            in_quote = !in_quote;
            continue;
        }
        if (c == ',' && !in_quote) {
            out.push_back(cur); cur.clear();
        } else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

void VocabBuilder::save_vocab(const string &out_dir) const
{
    // ensure out_dir exists (but do NOT create nested unnecessary folders; out_dir itself should exist or be created)
    try {
        if (!out_dir.empty()) fs::create_directories(fs::path(out_dir));
    } catch (...) {
        // best-effort
    }

    // tokens.csv: column,token,term_id,docfreq
    ofstream tok_out(fs::path(out_dir) / "tokens.csv");
    if (tok_out.is_open()) {
        tok_out << "column,token,term_id,docfreq\n";
        for (const auto &colpair : token2id_per_col) {
            const string &col = colpair.first;
            const auto &map_token2id = colpair.second;
            // invert token->id to produce stable order by id
            vector<pair<int,string>> id_token;
            id_token.reserve(map_token2id.size());
            for (auto &p : map_token2id) id_token.push_back(make_pair(p.second, p.first));
            sort(id_token.begin(), id_token.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
            for (auto &pr : id_token) {
                int tid = pr.first;
                const string &token = pr.second;
                int df = 0;
                auto df_it = docfreq_per_col.find(col);
                if (df_it != docfreq_per_col.end()) {
                    auto inner = df_it->second.find(tid);
                    if (inner != df_it->second.end()) df = inner->second;
                }
                tok_out << csv_escape(col) << "," << csv_escape(token) << "," << tid << "," << df << "\n";
            }
        }
        tok_out.close();
    }

    // clubs_map.csv
    ofstream clubs_out(fs::path(out_dir) / "clubs_map.csv");
    if (clubs_out.is_open()) {
        clubs_out << "club_id,slug,title\n";
        vector<pair<int,string>> id_slug;
        id_slug.reserve(club_to_id.size());
        for (auto it = club_to_id.begin(); it != club_to_id.end(); ++it) id_slug.push_back(make_pair(it->second, it->first));
        sort(id_slug.begin(), id_slug.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
        for (auto &p : id_slug) {
            int id = p.first;
            const string &slug = p.second;
            const string &title = (club_slug_to_title.find(slug) != club_slug_to_title.end()) ? club_slug_to_title.at(slug) : string();
            clubs_out << id << "," << csv_escape(slug) << "," << csv_escape(title) << "\n";
        }
        clubs_out.close();
    }

    // addresses_map.csv
    ofstream addr_out(fs::path(out_dir) / "addresses_map.csv");
    if (addr_out.is_open()) {
        addr_out << "address_id,address\n";
        vector<pair<int,string>> id_addr;
        id_addr.reserve(address_to_id.size());
        for (auto it = address_to_id.begin(); it != address_to_id.end(); ++it) id_addr.push_back(make_pair(it->second, it->first));
        sort(id_addr.begin(), id_addr.end(), [](const pair<int,string>& A, const pair<int,string>& B){ return A.first < B.first; });
        for (auto &p : id_addr) {
            addr_out << p.first << "," << csv_escape(p.second) << "\n";
        }
        addr_out.close();
    }
}

bool VocabBuilder::load_vocab(const string &in_dir)
{
    token2id_per_col.clear();
    docfreq_per_col.clear();
    club_to_id.clear();
    club_slug_to_title.clear();
    address_to_id.clear();

    // tokens.csv
    ifstream tok_in(fs::path(in_dir) / "tokens.csv");
    if (!tok_in.is_open()) return false;
    string header;
    if (!getline(tok_in, header)) return false;
    string line;
    while (getline(tok_in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line(line);
        if (cols.size() < 4) continue;
        string col = cols[0];
        string token = cols[1];
        int tid = atoi(cols[2].c_str());
        int df = atoi(cols[3].c_str());
        // ensure maps exist
        if (token2id_per_col.find(col) == token2id_per_col.end()) {
            token2id_per_col[col] = unordered_map<string,int>();
            docfreq_per_col[col] = unordered_map<int,int>();
        }
        token2id_per_col[col][token] = tid;
        docfreq_per_col[col][tid] = df;
    }
    tok_in.close();

    // clubs_map.csv
    ifstream clubs_in(fs::path(in_dir) / "clubs_map.csv");
    if (!clubs_in.is_open()) return false;
    if (!getline(clubs_in, header)) return false;
    while (getline(clubs_in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line(line);
        if (cols.size() < 3) continue;
        int id = atoi(cols[0].c_str());
        string slug = cols[1];
        string title = cols[2];
        // store by slug -> id and slug -> title
        club_to_id[slug] = id;
        club_slug_to_title[slug] = title;
    }
    clubs_in.close();

    // addresses_map.csv
    ifstream addr_in(fs::path(in_dir) / "addresses_map.csv");
    if (!addr_in.is_open()) return false;
    if (!getline(addr_in, header)) return false;
    while (getline(addr_in, line)) {
        if (line.empty()) continue;
        vector<string> cols = split_csv_line(line);
        if (cols.size() < 2) continue;
        int id = atoi(cols[0].c_str());
        string addr = cols[1];
        address_to_id[addr] = id;
    }
    addr_in.close();

    return true;
}
