#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include <vector>
#include <unordered_map>
#include <utility>
#include <string>

using namespace std;

struct UserProfile;

struct Recommender {
    Recommender(unordered_map<int, unordered_map<int,float>>* uf, unordered_map<int, vector<int>>* al);
    Recommender(unordered_map<int, UserProfile>* profiles_in, unordered_map<int, vector<int>>* al);

    vector<pair<int,float>> recommend_by_cosine(int user, int topk);
    vector<pair<int,float>> recommend_from_supernodes(int user, const unordered_map<int, unordered_map<int,float>>& super_feats, int topk);

    vector<pair<int,float>> recommend_by_profile(int user, int topk);

    float profile_similarity(const UserProfile &A, const UserProfile &B, const vector<string> &text_columns) const;

    void set_column_normalizers(const vector<pair<float,float>>& norms);
    void set_field_normalizers(const unordered_map<string, pair<float,float>>& m);

    unordered_map<int, unordered_map<int,float>>* user_feats = nullptr;
    unordered_map<int, vector<int>>* adj_list = nullptr;

    unordered_map<int, UserProfile>* profiles = nullptr;

    vector<pair<float,float>> column_normalizers;
    unordered_map<string, pair<float,float>> field_normalizers;
};

#endif
