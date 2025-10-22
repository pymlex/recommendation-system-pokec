#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H
#include <unordered_map>
#include <vector>
#include <utility>
#include <string>
using namespace std;
struct GraphBuilder {
    unordered_map<int, vector<pair<int,float>>> adjacency;
    void load_edges(const string& path, size_t max_lines = 0);
    vector<int> neighbors(int uid);
};
#endif
