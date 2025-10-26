#include "graph_builder.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

void GraphBuilder::load_edges(const string& path, size_t max_lines) {
    ifstream in(path);
    string line;
    size_t cnt = 0;
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        int a = 0, b = 0;
        ss >> a >> b;
        adjacency[a].push_back(make_pair(b, 1.0f));
        ++cnt;
        if (max_lines && cnt >= max_lines) break;
    }
}

vector<int> GraphBuilder::neighbors(int uid) {
    vector<int> out;
    auto it = adjacency.find(uid);
    if (it == adjacency.end()) return out;
    for (auto &p : it->second) out.push_back(p.first);
    return out;
}

static inline string trim_copy_g(const string& s) {
    size_t a = 0;
    while (a < s.size() && isspace((unsigned char)s[a])) ++a;
    size_t b = s.size();
    while (b > a && isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b - a);
}

bool GraphBuilder::load_serialized(const string& path) {
    adjacency.clear();
    ifstream in(path);
    if (!in.is_open()) return false;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        bool first = true;
        int uid = -1;
        while (getline(ss, token, ',')) {
            string t = trim_copy_g(token);
            if (t.empty()) continue;
            if (first) { uid = atoi(t.c_str()); first = false; continue; }
            int nid = atoi(t.c_str());
            adjacency[uid].push_back(make_pair(nid, 1.0f));
        }
    }
    return true;
}

bool GraphBuilder::save_serialized(const string& path) const {
    ofstream out(path);
    if (!out.is_open()) return false;
    vector<int> keys;
    keys.reserve(adjacency.size());
    for (auto it = adjacency.begin(); it != adjacency.end(); ++it) keys.push_back(it->first);
    sort(keys.begin(), keys.end());
    for (int uid : keys) {
        out << uid;
        const auto &vec = adjacency.at(uid);
        for (auto &p : vec) out << "," << p.first;
        out << "\n";
    }
    return true;
}
