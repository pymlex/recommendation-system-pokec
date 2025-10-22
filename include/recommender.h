#ifndef RECOMMENDER_H
#define RECOMMENDER_H
#include <vector>
#include <unordered_map>
using namespace std;
struct Recommender {
    Recommender(unordered_map<int, unordered_map<int,float>>* uf, unordered_map<int, vector<int>>* al);
    vector<pair<int,float>> recommend_by_cosine(int user, int topk);
    vector<pair<int,float>> recommend_from_supernodes(int user, const unordered_map<int, unordered_map<int,float>>& super_feats, int topk);
    unordered_map<int, unordered_map<int,float>>* user_feats;
    unordered_map<int, vector<int>>* adj_list;
};
#endif
