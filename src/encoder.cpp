#include "encoder.h"
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>


using namespace std;


Encoder::Encoder(
    const vector<string>& colKeys_in,
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col_in,
    const unordered_map<string,int>& club_to_id_in)
    : colKeys(colKeys_in),
      token2id_per_col(token2id_per_col_in),
      club_to_id(club_to_id_in)
{
    region_to_id.clear();
}

vector<string> Encoder::split_line_to_cols(const string& line) const
{
    vector<string> cols;
    stringstream ss(line);
    string cell;

    while (getline(ss, cell, '\t'))
        cols.push_back(cell);

    return cols;
}

unordered_map<int,int> Encoder::extract_club_counts_from_line(const string& line, const std::regex& href_re) const
{
    unordered_map<int,int> club_counts;
    string::const_iterator start = line.cbegin();
    smatch m;

    while (regex_search(start, line.cend(), m, href_re))
    {
        string slug = m[1].str();
        auto it = club_to_id.find(slug);

        if (it != club_to_id.end())
            club_counts[it->second] += 1;

        start = m.suffix().first;
    }

    return club_counts;
}

string Encoder::format_counts_to_csv(const unordered_map<int,int>& counts) const
{
    string out;
    bool first = true;

    for (auto it = counts.begin(); it != counts.end(); ++it)
    {
        if (! first)
            out.push_back(';');
        first = false;
        out += to_string(it->first);
    }

    return out;
}

string Encoder::format_token_counts_to_csv(const unordered_map<int,int>& counts) const
{
    string out;
    bool first = true;

    for (auto it = counts.begin(); it != counts.end(); ++it)
    {
        if (! first)
            out.push_back(';');
        first = false;
        out += to_string(it->first);
        out.push_back(':');
        out += to_string(it->second);
    }

    return out;
}

unordered_map<int,int> Encoder::encode_tokens_for_column(
    const string& text,
    const string& key,
    Tokenizer& tok,
    Lemmatiser& lem) const
{
    unordered_map<int,int> counts;

    if (text.empty())
        return counts;

    vector<string> toks = tok.tokenize(text);
    vector<string> lems = lem.lemmatize_tokens(toks);

    auto itmap = token2id_per_col.find(key);
    if (itmap == token2id_per_col.end())
        return counts;

    const unordered_map<string,int>& mapid = itmap->second;

    for (size_t j = 0; j < lems.size(); ++j)
    {
        const string& w = lems[j];
        auto it3 = mapid.find(w);
        if (it3 != mapid.end())
            counts[it3->second] += 1;
    }

    return counts;
}

int Encoder::get_or_add_region_id(const string& region)
{
    if (region.empty())
        return -1;

    auto it = region_to_id.find(region);
    if (it != region_to_id.end())
        return it->second;

    int nid = (int) region_to_id.size();
    region_to_id[region] = nid;
    return nid;
}

void Encoder::pass2(const string& profiles_tsv, const string& out_users_csv)
{
    ifstream in(profiles_tsv);
    ofstream out(out_users_csv);

    if (! in.is_open()) {
        cerr << "[Encoder] cannot open input file: " << profiles_tsv << "\n";
        return;
    }

    if (! out.is_open()) {
        cerr << "[Encoder] cannot open output file: " << out_users_csv << "\n";
        return;
    }

    out << "user_id,gender,region,age,clubs";
    for (size_t i = 0; i < colKeys.size(); ++i)
        out << "," << colKeys[i] << "_tokens";
    out << "\n";

    string line;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>");
    Tokenizer tok;
    Lemmatiser lemma("data/lem-me-sk.bin");

    while (getline(in, line))
    {
        if (line.empty())
            continue;

        vector<string> cols = split_line_to_cols(line);

        string user_id = cols.size() > 0 ? cols[0] : "";
        string gender  = cols.size() > 3 ? cols[3] : "";
        string region  = cols.size() > 4 ? cols[4] : "";
        string age     = cols.size() > 7 ? cols[7] : "";

        unordered_map<int,int> club_counts = extract_club_counts_from_line(line, href_re);
        string clubs_csv = format_counts_to_csv(club_counts);

        int region_id = get_or_add_region_id(region);
        string region_field = (region_id >= 0) ? to_string(region_id) : "";

        out << user_id << "," << gender << "," << region_field << "," << age << "," << clubs_csv;

        for (size_t i = 0; i < colKeys.size(); ++i)
        {
            string key = colKeys[i];
            size_t idx = 10 + i;
            string text = idx < cols.size() ? cols[idx] : "";

            unordered_map<int,int> counts = encode_tokens_for_column(text, key, tok, lemma);
            string tok_csv = format_token_counts_to_csv(counts);
            out << "," << tok_csv;
        }

        out << "\n";
    }

    out.close();
    in.close();
}

void Encoder::save_addresses_map(const string& out_csv) const
{
    ofstream out(out_csv);
    if (! out.is_open()) {
        cerr << "[Encoder] cannot open addresses map file for writing: " << out_csv << "\n";
        return;
    }

    out << "region_id,region\n";

    unordered_map<int,string> id2region;
    for (auto it = region_to_id.begin(); it != region_to_id.end(); ++it)
        id2region[it->second] = it->first;

    for (int i = 0; i < (int) id2region.size(); ++i)
    {
        const string& r = id2region[i];
        bool need_quote = (r.find(',') != string::npos) || (r.find('"') != string::npos);

        out << i << ",";
        if (need_quote) out << '"';
        for (size_t k = 0; k < r.size(); ++k)
        {
            if (r[k] == '"') out << "\"\"";
            else out << r[k];
        }
        if (need_quote) out << '"';
        out << "\n";
    }

    out.close();
}
