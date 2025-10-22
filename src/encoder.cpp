#include "encoder.h"
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <regex>
using namespace std;

Encoder::Encoder(const vector<string>& colKeys_in,
                 const unordered_map<string, unordered_map<string,int>>& token2id_per_col_in,
                 const unordered_map<string,int>& club_to_id_in)
    : colKeys(colKeys_in),
      token2id_per_col(token2id_per_col_in),
      club_to_id(club_to_id_in)
{
}

void Encoder::pass2(const string& profiles_tsv, const string& out_users_csv)
{
    ifstream in(profiles_tsv);
    ofstream out(out_users_csv);
    out << "user_id,gender,region,age,clubs";
    for (size_t i = 0; i < colKeys.size(); ++i) out << "," << colKeys[i] << "_tokens";
    out << "\n";
    string line;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>");
    Tokenizer tok;
    Lemmatiser lemma("data/lem-me-sk.bin");
    while (getline(in, line))
    {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<string> cols;
        string cell;
        while (getline(ss, cell, '\t')) cols.push_back(cell);
        string user_id = cols.size() > 0 ? cols[0] : "";
        string gender = cols.size() > 3 ? cols[3] : "";
        string region = cols.size() > 4 ? cols[4] : "";
        string age = cols.size() > 7 ? cols[7] : "";
        unordered_map<int,int> club_counts;
        string::const_iterator start = line.cbegin();
        smatch m;
        while (regex_search(start, line.cend(), m, href_re))
        {
            string slug = m[1].str();
            auto it = club_to_id.find(slug);
            if (it != club_to_id.end()) club_counts[it->second] += 1;
            start = m.suffix().first;
        }
        string clubs_csv;
        bool first = true;
        for (auto it2 = club_counts.begin(); it2 != club_counts.end(); ++it2)
        {
            if (! first) clubs_csv.push_back(';');
            first = false;
            clubs_csv += to_string(it2->first);
        }
        out << user_id << "," << gender << "," << region << "," << age << "," << clubs_csv;
        for (size_t i = 0; i < colKeys.size(); ++i)
        {
            string key = colKeys[i];
            size_t idx = 10;
            string text = idx < cols.size() ? cols[idx] : "";
            vector<string> toks = tok.tokenize(text);
            vector<string> lems = lemma.lemmatize_tokens(toks);
            unordered_map<int,int> counts;
            auto itmap = token2id_per_col.find(key);
            if (itmap != token2id_per_col.end())
            {
                const unordered_map<string,int>& mapid = itmap->second;
                for (size_t j = 0; j < lems.size(); ++j)
                {
                    const string& w = lems[j];
                    auto it3 = mapid.find(w);
                    if (it3 != mapid.end()) counts[it3->second] += 1;
                }
            }
            string tok_csv;
            bool f2 = true;
            for (auto itc = counts.begin(); itc != counts.end(); ++itc)
            {
                if (! f2) tok_csv.push_back(';');
                f2 = false;
                tok_csv += to_string(itc->first);
                tok_csv.push_back(':');
                tok_csv += to_string(itc->second);
            }
            out << "," << tok_csv;
        }
        out << "\n";
    }
}
