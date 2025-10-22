#ifndef HIERCOARSENER_H
#define HIERCOARSENER_H

#include <unordered_map>
#include <vector>

using namespace std;

struct HierCoarsener {
public:
    HierCoarsener(int max_super_size = 100, float size_penalty_in = 0.5f);

    unordered_map<int, int> node_to_super;
    unordered_map<int, unordered_map<int,float>> super_features;
    unordered_map<int, vector<int>> super_members;

    void coarsen(const unordered_map<int, unordered_map<int,float>>& user_feats,
                 const unordered_map<int, vector<int>>& adj_list,
                 int levels);

private:
    int max_supernode_size;
    float size_penalty;

    void coarsen_level(const unordered_map<int, unordered_map<int,float>>& feats,
                       const unordered_map<int, vector<int>>& adj_list,
                       const unordered_map<int,int>& sizes);
};

#endif
