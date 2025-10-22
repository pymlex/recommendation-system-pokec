#include "vocab_builder.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>

using namespace std;

VocabBuilder::VocabBuilder(const vector<string> &colKeys_in)
{
    colKeys = colKeys_in;
    for (const auto &key : colKeys) {
        token2id_per_col[key] = unordered_map<string,int>();
        docfreq_per_col[key] = unordered_map<int,int>();
    }
    club_to_id.clear();
}

void VocabBuilder::pass1(const string &profiles_tsv, Tokenizer &tok, Lemmatiser &lem)
{
    ifstream in(profiles_tsv);
    if (!in.is_open()) {
        cerr << "[VocabBuilder] cannot open " << profiles_tsv << "\n";
        return;
    }

    string line;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>([^<]*)</a>");

    while (getline(in, line)) {
        if (line.empty()) continue;

        vector<string> cols;
        string cell;
        stringstream ss(line);
        while (getline(ss, cell, '\t')) cols.push_back(cell);

        string::const_iterator start = line.cbegin();
        smatch m;
        while (regex_search(start, line.cend(), m, href_re)) {
            string raw_slug = m[1].str();
            string slug;
            for (size_t i = 0; i < raw_slug.size(); ++i) {
                unsigned char c = (unsigned char) raw_slug[i];
                if (c >= 'A' && c <= 'Z') slug.push_back((char)(c + ('a' - 'A')));
                else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || raw_slug[i] == '-') slug.push_back(raw_slug[i]);
                else if (!slug.empty() && slug.back() != '-') slug.push_back('-');
            }
            while (!slug.empty() && slug.back() == '-') slug.pop_back();
            if (!slug.empty() && club_to_id.find(slug) == club_to_id.end()) {
                int nid = (int) club_to_id.size();
                club_to_id[slug] = nid;
            }
            start = m.suffix().first;
        }

        for (size_t ci = 0; ci < colKeys.size(); ++ci) {
            const string &key = colKeys[ci];
            size_t idx = 10 + ci;
            if (idx >= cols.size()) continue;
            string text = cols[idx];

            vector<string> tokens = tok.tokenize(text);
            vector<string> lem_tokens = lem.lemmatize_tokens(tokens);

            for (const string &t : lem_tokens) {
                if (token2id_per_col[key].find(t) == token2id_per_col[key].end()) {
                    int nid = (int)token2id_per_col[key].size();
                    token2id_per_col[key][t] = nid;
                }
                int tid = token2id_per_col[key][t];
                docfreq_per_col[key][tid] += 1;
            }
        }
    }
}

void VocabBuilder::save_vocab(const std::string &filename)
{
    std::ofstream out(filename);
    if (!out.is_open()) return;

    for (const auto &col : token2id_per_col) {
        const std::string &col_name = col.first;
        out << "# Column: " << col_name << "\n";
        for (const auto &p : col.second) {
            out << p.first << "\t" << p.second << "\n";
        }
    }
}