#include "preprocess.h"
#include <fstream>
#include <sstream>
#include <regex>
using namespace std;

vector<string> split_tab(const string& line)
{
    vector<string> out;
    string field;
    stringstream ss(line);
    while (getline(ss, field, '\t')) out.push_back(field);
    return out;
}

vector<vector<string>> preprocess_profiles(const string& path, Tokenizer& tok, size_t max_rows)
{
    ifstream in(path);
    vector<vector<string>> df;
    string line;
    size_t row = 0;
    regex href_re("href=\"/klub/([^\"]+)\"");
    while (getline(in, line))
    {
        if (line.size() == 0) continue;
        vector<string> cols = split_tab(line);
        if (cols.size() == 0) continue;
        vector<string> out;
        if (cols.size() >= 1) out.push_back(cols[0]);
        if (cols.size() >= 4) out.push_back(cols[3]);
        for (size_t i = 10; i < cols.size(); ++i)
        {
            string cell = cols[i];
            if (cell.find("<a ") != string::npos || cell.find("klub") != string::npos)
            {
                smatch m;
                string s = cell;
                string::const_iterator searchStart(s.cbegin());
                string res;
                while (regex_search(searchStart, s.cend(), m, href_re))
                {
                    if (res.size()) res.push_back(' ');
                    string token = m[1].str();
                    for (size_t k = 0; k < token.size(); ++k)
                    {
                        char c = token[k];
                        unsigned char uc = (unsigned char) c;
                        if ( (uc >= '0' && uc <= '9') || (uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z') || c == '-' )
                        {
                            if (uc >= 'A' && uc <= 'Z') res.push_back((char)(uc + 32));
                            else res.push_back(c);
                        }
                        else
                        {
                            if (res.empty() == false && res.back() != '-') res.push_back('-');
                        }
                    }
                    searchStart = m.suffix().first;
                }
                if (res.size() == 0)
                {
                    vector<string> toks = tok.tokenize(cell);
                    string joined;
                    for (size_t k = 0; k < toks.size(); ++k)
                    {
                        if (k) joined.push_back(' ');
                        joined += toks[k];
                    }
                    out.push_back(joined);
                }
                else out.push_back(res);
            }
            else
            {
                vector<string> toks = tok.tokenize(cell);
                string joined;
                for (size_t k = 0; k < toks.size(); ++k)
                {
                    if (k) joined.push_back(' ');
                    joined += toks[k];
                }
                out.push_back(joined);
            }
        }
        df.push_back(out);
        ++row;
        if (max_rows && row >= max_rows) break;
    }
    return df;
}

void save_df_csv(const string& outpath, const vector<vector<string>>& df)
{
    ofstream out(outpath);
    for (size_t r = 0; r < df.size(); ++r)
    {
        const vector<string>& row = df[r];
        for (size_t c = 0; c < row.size(); ++c)
        {
            string cell = row[c];
            bool need_quote = (cell.find(',') != string::npos) || (cell.find('"') != string::npos);
            if (need_quote) out << '"';
            for (size_t i = 0; i < cell.size(); ++i)
            {
                char ch = cell[i];
                if (ch == '"') out << "\"\""; else out << ch;
            }
            if (need_quote) out << '"';
            if (c + 1 < row.size()) out << ',';
        }
        out << '\n';
    }
}
