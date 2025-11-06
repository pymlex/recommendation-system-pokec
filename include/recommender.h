#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include <vector>
#include <unordered_map>
#include <utility>
#include <string>

struct UserProfile;
struct TFIDFIndex;

struct Recommender {
    Recommender(std::unordered_map<int, std::unordered_map<int,float>>* uf, std::unordered_map<int, std::vector<int>>* al);
    Recommender(std::unordered_map<int, UserProfile>* profiles_in, std::unordered_map<int, std::vector<int>>* al);

    std::vector<std::pair<int,float>> recommend_by_cosine(int user, int topk);
    std::vector<std::pair<int,float>> recommend_from_supernodes(int user, const std::unordered_map<int, std::unordered_map<int,float>>& super_feats, int topk);

    std::vector<std::pair<int,float>> recommend_by_profile(int user, int topk);

    float profile_similarity(const UserProfile &A, const UserProfile &B, const std::vector<std::string> &text_columns) const;

    void set_column_normalizers(const std::vector<std::pair<float,float>>& norms);
    void set_field_normalizers(const std::unordered_map<std::string, std::pair<float,float>>& m);
    void set_text_columns(const std::vector<std::string>& cols);
    void set_tfidf_index(TFIDFIndex* idx);

    // friend recommenders
    std::vector<std::pair<int,float>> recommend_friends_graph(int user, int topk, int max_candidates = 10000);
    std::vector<std::pair<int,float>> recommend_friends_collab(int user, int topk, int max_candidates = 10000);
    std::vector<std::pair<int,float>> recommend_friends_by_interest(int user, int topk, int max_candidates = 10000);

    std::unordered_map<int, std::unordered_map<int,float>>* user_feats = nullptr;
    std::unordered_map<int, std::vector<int>>* adj_list = nullptr;

    std::unordered_map<int, UserProfile>* profiles = nullptr;

    std::vector<std::pair<float,float>> column_normalizers;
    std::unordered_map<std::string, std::pair<float,float>> field_normalizers;

    std::vector<std::string> text_columns;
    TFIDFIndex* tfidf = nullptr;
};

#endif
