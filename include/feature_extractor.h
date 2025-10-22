#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
struct FeatureExtractor {
    unordered_map<string, int> vocab;
    unordered_map<int, unordered_map<int, float>> user_tfidf;
    void build_from_df(const vector<vector<string>>& df);
    vector<pair<int,float>> cosine_sim_sparse(const unordered_map<int,float>& a, int topk);
    float cosine_between(const unordered_map<int,float>& a, const unordered_map<int,float>& b);
};
#endif
