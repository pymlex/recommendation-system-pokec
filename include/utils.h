#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

unordered_map<int, vector<int>> build_adj_list(const unordered_map<int, vector<pair<int,float>>>& adj_weighted);
float evaluate_holdout_hit_at_k(const unordered_map<int, vector<int>>& adj_list,
                                const unordered_map<int, unordered_map<int,float>>& feats,
                                int sample_size, int k);
vector<string> load_text_columns_from_file(const string& path);

#endif
