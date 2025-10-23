#include "graph_builder.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

void GraphBuilder::load_edges(const string& path, size_t max_lines)
{
    ifstream in(path);
    string line;
    size_t cnt = 0;

    while (getline(in, line))
    {
        if (line.size() == 0)
            continue;
        stringstream ss(line);
        int a, b;
        ss >> a >> b;
        adjacency[a].push_back(make_pair(b, 1.0f));
        ++cnt;
        if (max_lines && cnt >= max_lines)
            break;
    }
}

vector<int> GraphBuilder::neighbors(int uid)
{
    vector<int> out;
    if (adjacency.find(uid) == adjacency.end())
        return out;
    const vector<pair<int,float>>& vec = adjacency[uid];
    for (size_t i = 0; i < vec.size(); ++i)
        out.push_back(vec[i].first);
    return out;
}

static inline string trim_copy(const string& s)
{
    size_t a = 0;
    while (a < s.size() && isspace((unsigned char)s[a])) ++a;
    size_t b = s.size();
    while (b > a && isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b - a);
}

bool GraphBuilder::load_serialized(const string& path)
{
    adjacency.clear();
    ifstream in(path);
    if (! in.is_open())
        return false;

    string line;
    while (getline(in, line))
    {
        if (line.size() == 0)
            continue;

        stringstream ss(line);
        string token;
        bool first = true;
        int uid = -1;
        while (getline(ss, token, ','))
        {
            string t = trim_copy(token);
            if (t.empty())
                continue;
            if (first)
            {
                uid = atoi(t.c_str());
                first = false;
                continue;
            }
            int nid = atoi(t.c_str());
            adjacency[uid].push_back(make_pair(nid, 1.0f));
        }
    }
    return true;
}

bool GraphBuilder::save_serialized(const string& path) const
{
    ofstream out(path);
    if (! out.is_open())
        return false;

    for (auto it = adjacency.begin(); it != adjacency.end(); ++it)
    {
        int uid = it->first;
        out << uid;
        const vector<pair<int,float>>& vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i)
        {
            out << "," << vec[i].first;
        }
        out << "\n";
    }
    return true;
}
