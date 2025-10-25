#include "encoder.h"
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>

using namespace std;

struct EncRow {
    string user_id_str;
    int user_id;
    string gender;
    int region_id;
    string age;
    string clubs_csv;
    string friends_csv;
    vector<string> token_cols_csv;
};

Encoder::Encoder(
    const vector<string>& colKeys_in,
    const unordered_map<string, unordered_map<string,int>>& token2id_per_col_in,
    const unordered_map<string,int>& club_to_id_in,
    const unordered_map<string,int>& address_to_id_in,
    const unordered_map<int, vector<int>>& adjacency_in)
    : colKeys(colKeys_in),
      token2id_per_col(token2id_per_col_in),
      club_to_id(club_to_id_in),
      address_to_id(address_to_id_in),
      adjacency(adjacency_in)
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
        string raw = m[1].str();
        string slug;
        slug.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
        {
            unsigned char c = (unsigned char) raw[i];
            if (c >= 'A' && c <= 'Z') slug.push_back((char)(c + ('a' - 'A')));
            else slug.push_back(raw[i]);
        }
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
    if (counts.empty()) return string();
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
    if (text.empty()) return counts;
    vector<string> toks = tok.tokenize(text);
    vector<string> lems = lem.lemmatize_tokens(toks);
    auto itmap = token2id_per_col.find(key);
    if (itmap == token2id_per_col.end()) return counts;
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

int Encoder::lookup_region_id(const string& region) const
{
    if (region.empty()) return -1;
    string norm;
    norm.reserve(region.size());
    for (size_t i = 0; i < region.size(); ++i)
    {
        unsigned char c = (unsigned char) region[i];
        if (c >= 'A' && c <= 'Z') norm.push_back((char)(c + ('a' - 'A')));
        else norm.push_back(region[i]);
    }
    while (!norm.empty() && (norm.back() == '\r' || norm.back() == '\n' || norm.back() == '\t')) norm.pop_back();
    while (!norm.empty() && (norm.front() == '\t' || norm.front() == ' ')) norm.erase(norm.begin());
    auto it = address_to_id.find(norm);
    if (it == address_to_id.end()) return -1;
    return it->second;
}

static vector<int> adjacency_get_friends(const unordered_map<int, vector<int>>& adj, int uid)
{
    auto it = adj.find(uid);
    if (it == adj.end()) return vector<int>();
    return it->second;
}

void Encoder::pass2(const string& profiles_tsv, const string& out_users_csv)
{
    ifstream in(profiles_tsv);
    vector<EncRow> rows;
    rows.reserve(100000);

    string line;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>");
    Tokenizer tok;
    Lemmatiser lem("data/lem-me-sk.bin");

    while (getline(in, line))
    {
        if (line.empty()) continue;

        vector<string> cols = split_line_to_cols(line);

        EncRow er;
        er.user_id_str = cols.size() > 0 ? cols[0] : "";
        er.user_id = atoi(er.user_id_str.c_str());
        er.gender = cols.size() > 3 ? cols[3] : "";
        string region = cols.size() > 4 ? cols[4] : "";
        er.age = cols.size() > 7 ? cols[7] : "";
        er.region_id = lookup_region_id(region);

        unordered_map<int,int> club_counts = extract_club_counts_from_line(line, href_re);
        er.clubs_csv = format_counts_to_csv(club_counts);

        vector<int> friends = adjacency_get_friends(adjacency, er.user_id);
        if (friends.empty()) er.friends_csv = string();
        else {
            bool first = true;
            string out;
            for (size_t i = 0; i < friends.size(); ++i)
            {
                if (! first) out.push_back(';');
                first = false;
                out += to_string(friends[i]);
            }
            er.friends_csv = out;
        }

        er.token_cols_csv.resize(colKeys.size());
        for (size_t i = 0; i < colKeys.size(); ++i)
        {
            string key = colKeys[i];
            size_t idx = 10 + i;
            string text = idx < cols.size() ? cols[idx] : "";
            if (text.empty()) {
                er.token_cols_csv[i] = string();
                continue;
            }
            unordered_map<int,int> counts = encode_tokens_for_column(text, key, tok, lem);
            er.token_cols_csv[i] = format_token_counts_to_csv(counts);
        }

        rows.push_back(std::move(er));
    }

    sort(rows.begin(), rows.end(), [](const EncRow& A, const EncRow& B){ return A.user_id < B.user_id; });

    ofstream out(out_users_csv);
    out << "user_id,gender,region_id,age,clubs,friends";
    for (size_t i = 0; i < colKeys.size(); ++i)
        out << "," << colKeys[i] << "_tokens";
    out << "\n";

    for (size_t r = 0; r < rows.size(); ++r)
    {
        const EncRow& er = rows[r];
        out << er.user_id_str << "," << er.gender << ",";
        if (er.region_id >= 0) out << er.region_id; else out << "";
        out << "," << er.age << "," << er.clubs_csv << "," << er.friends_csv;
        for (size_t i = 0; i < er.token_cols_csv.size(); ++i)
            out << "," << er.token_cols_csv[i];
        out << "\n";
    }
}
