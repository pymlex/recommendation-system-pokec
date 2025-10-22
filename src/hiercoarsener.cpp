#include "hiercoarsener.h"
#include <cmath>
#include <algorithm>
#include <set>

using namespace std;

HierCoarsener::HierCoarsener(int max_super_size, float size_penalty_in) {
    max_supernode_size = max_super_size;
    size_penalty = size_penalty_in;
}

void HierCoarsener::coarsen_level(const unordered_map<int, unordered_map<int,float>>& feats,
                                  const unordered_map<int, vector<int>>& adj_list,
                                  const unordered_map<int,int>& sizes) {
    node_to_super.clear();
    super_features.clear();
    super_members.clear();

    set<int> visited;
    int next_super = 0;

    for (auto it = feats.begin(); it != feats.end(); ++it) {
        int u = it->first;
        if (visited.find(u) != visited.end()) continue;
        visited.insert(u);

        int best_v = -1;
        float best_score = 0.0f;

        int size_u = 1;
        if (sizes.find(u) != sizes.end()) size_u = sizes.at(u);

        if (adj_list.find(u) != adj_list.end()) {
            const vector<int>& neigh = adj_list.at(u);
            for (size_t i = 0; i < neigh.size(); ++i) {
                int v = neigh[i];
                if (visited.find(v) != visited.end()) continue;
                if (feats.find(v) == feats.end()) continue;

                int size_v = 1;
                if (sizes.find(v) != sizes.end()) size_v = sizes.at(v);

                int total_size = size_u + size_v;
                if (max_supernode_size > 0 && total_size > max_supernode_size) continue;

                const unordered_map<int,float>& A = it->second;
                const unordered_map<int,float>& B = feats.at(v);

                float dot = 0.0f;
                if (A.size() < B.size()) {
                    for (auto ait = A.begin(); ait != A.end(); ++ait) {
                        int k = ait->first;
                        float va = ait->second;
                        auto jt = B.find(k);
                        if (jt != B.end()) dot += va * jt->second;
                    }
                }
                else {
                    for (auto bit = B.begin(); bit != B.end(); ++bit) {
                        int k = bit->first;
                        float vb = bit->second;
                        auto jt = A.find(k);
                        if (jt != A.end()) dot += vb * jt->second;
                    }
                }

                float penalty = 0.0f;
                if (max_supernode_size > 0) {
                    float frac = (float) (total_size - 1) / (float) max_supernode_size;
                    if (frac < 0.0f) frac = 0.0f;
                    if (frac > 1.0f) frac = 1.0f;
                    penalty = size_penalty * frac;
                }

                float score = dot * (1.0f - penalty);

                if (score > best_score) {
                    best_score = score;
                    best_v = v;
                }
            }
        }

        if (best_v != -1 && best_score > 0.0f) {
            visited.insert(best_v);

            node_to_super[u] = next_super;
            node_to_super[best_v] = next_super;

            int size_v = 1;
            if (sizes.find(best_v) != sizes.end()) size_v = sizes.at(best_v);
            int total_size = size_u + size_v;

            unordered_map<int,float> merged;
            const unordered_map<int,float>& A = it->second;
            const unordered_map<int,float>& B = feats.at(best_v);

            for (auto ait = A.begin(); ait != A.end(); ++ait) {
                int k = ait->first;
                float va = ait->second;
                merged[k] += va * (float) size_u;
            }
            for (auto bit = B.begin(); bit != B.end(); ++bit) {
                int k = bit->first;
                float vb = bit->second;
                merged[k] += vb * (float) size_v;
            }

            for (auto mit = merged.begin(); mit != merged.end(); ++mit) mit->second /= (float) total_size;

            float sum2 = 0.0f;
            for (auto mit = merged.begin(); mit != merged.end(); ++mit) sum2 += mit->second * mit->second;
            float norm = sqrt(sum2);
            if (norm > 0.0f) {
                for (auto mit = merged.begin(); mit != merged.end(); ++mit) mit->second /= norm;
            }

            super_features[next_super] = merged;
            vector<int> members;
            members.push_back(u);
            members.push_back(best_v);
            super_members[next_super] = members;

            ++next_super;
        }
        else {
            node_to_super[u] = next_super;
            super_features[next_super] = it->second;
            vector<int> members;
            members.push_back(u);
            super_members[next_super] = members;
            ++next_super;
        }
    }
}

void HierCoarsener::coarsen(const unordered_map<int, unordered_map<int,float>>& user_feats,
                            const unordered_map<int, vector<int>>& adj_list,
                            int levels) {
    unordered_map<int, unordered_map<int,float>> current_feats = user_feats;
    unordered_map<int, vector<int>> current_adj = adj_list;
    unordered_map<int,int> sizes;

    for (auto it = current_feats.begin(); it != current_feats.end(); ++it) sizes[it->first] = 1;

    for (int l = 0; l < levels; ++l) {
        coarsen_level(current_feats, current_adj, sizes);

        unordered_map<int, unordered_map<int,float>> next_feats;
        unordered_map<int, vector<int>> next_adj;
        unordered_map<int,int> next_sizes;

        for (auto it = super_features.begin(); it != super_features.end(); ++it) {
            next_feats[it->first] = it->second;
            next_sizes[it->first] = (int) super_members[it->first].size();
        }

        for (auto it = current_adj.begin(); it != current_adj.end(); ++it) {
            int u = it->first;
            int su = node_to_super[u];
            const vector<int>& neigh = it->second;
            for (size_t i = 0; i < neigh.size(); ++i) {
                int v = neigh[i];
                int sv = node_to_super[v];
                if (su == sv) continue;
                next_adj[su].push_back(sv);
            }
        }

        for (auto it = next_adj.begin(); it != next_adj.end(); ++it) {
            vector<int>& vec = it->second;
            sort(vec.begin(), vec.end());
            vec.erase(unique(vec.begin(), vec.end()), vec.end());
        }

        current_feats = next_feats;
        current_adj = next_adj;
        sizes = next_sizes;
    }

    super_features = current_feats;
}
