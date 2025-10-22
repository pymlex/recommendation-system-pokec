#ifndef INCREMENTAL_BUILDER_H
#define INCREMENTAL_BUILDER_H
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
bool build_user_feats_from_bin(const string& bin_path, const unordered_map<int, pair<uint32_t,uint32_t> >& idx_map, const vector<int>& user_ids, unordered_map<int, unordered_map<int,float> >& out_feats);
#endif
