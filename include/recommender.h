#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;

struct UserProfile;

struct Recommender {
    // TF-IDF based constructor
    Recommender(unordered_map<int, unordered_map<int,float>>* uf,
                unordered_map<int, vector<int>>* al);

    // Profile-based constructor
    Recommender(unordered_map<int, UserProfile>* profiles_in,
                unordered_map<int, vector<int>>* al);

    vector<pair<int,float>> recommend_by_cosine(int user, int topk);
    vector<pair<int,float>> recommend_from_supernodes(int user,
        const unordered_map<int, unordered_map<int,float>>& super_feats, int topk);
    vector<pair<int,float>> recommend_by_profile(int user, int topk);

private:
    unordered_map<int, unordered_map<int,float>>* user_feats = nullptr;
    unordered_map<int, UserProfile>* profiles = nullptr;
    unordered_map<int, vector<int>>* adj_list = nullptr;

    float profile_similarity(const UserProfile& a, const UserProfile& b) const;
};

#endif
