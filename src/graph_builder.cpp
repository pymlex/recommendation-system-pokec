#include "graph_builder.h"
#include <fstream>
#include <sstream>
using namespace std;

void GraphBuilder::load_edges(const string& path, size_t max_lines)
{
    ifstream in(path);
    string line;
    size_t cnt = 0;
    while (getline(in, line))
    {
        if (line.size() == 0) continue;
        stringstream ss(line);
        int a, b;
        ss >> a >> b;
        adjacency[a].push_back(make_pair(b, 1.0f));
        ++cnt;
        if (max_lines && cnt >= max_lines) break;
    }
}

vector<int> GraphBuilder::neighbors(int uid)
{
    vector<int> out;
    if (adjacency.find(uid) == adjacency.end()) return out;
    const vector<pair<int,float>>& vec = adjacency[uid];
    for (size_t i = 0; i < vec.size(); ++i) out.push_back(vec[i].first);
    return out;
}
