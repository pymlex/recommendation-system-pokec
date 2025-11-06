#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <array>
#include <cstdint>

struct UserProfile;

struct RecommenderInternalGraph;
struct RecommenderInternalClubs;
struct RecommenderInternalSim;

class Recommender {
public:
    Recommender(const std::unordered_map<int, UserProfile>* profiles_in,
                const std::unordered_map<int, std::vector<int>>* al);
    Recommender(const std::unordered_map<int, std::unordered_map<int,float>>* user_feats_in,
                const std::unordered_map<int, std::vector<int>>* al);

    std::vector<std::pair<int,float>> recommend_graph_registration(int user, int topk, int candidate_limit = 10000) const;
    std::vector<std::pair<int,float>> recommend_collaborative(int user, int topk, int candidate_limit = 10000) const;
    std::vector<std::pair<int,float>> recommend_by_interest(int user, int topk, int candidate_limit = 10000) const;

    std::vector<std::pair<int,float>> recommend_clubs_collab(int user, int topk, int candidate_limit = 10000) const;

    void set_text_columns(const std::vector<std::string>& cols);
    void set_tfidf_index(const std::unordered_map<std::string, std::unordered_map<int,float>>& idf_map);

    std::vector<std::pair<int,float>> recommend_friends_graph(int user, int topk, int candidate_limit = 10000) const;
    std::vector<std::pair<int,float>> recommend_friends_collab(int user, int topk, int candidate_limit = 10000) const;
    std::vector<std::pair<int,float>> recommend_friends_by_interest(int user, int topk, int candidate_limit = 10000) const;
    std::vector<std::pair<int,float>> recommend_from_supernodes(int user, const std::unordered_map<int, std::unordered_map<int,float>>& super_feats, int topk) const;

    void set_field_normalizers(const std::unordered_map<std::string, std::pair<float,float>>& m);
    void set_column_normalizers(const std::unordered_map<std::string, std::pair<float,float>>& m);

    float profile_similarity(const UserProfile &A, const UserProfile &B, const std::vector<std::string> &text_columns) const;
    float profile_similarity(const UserProfile &A, const UserProfile &B) const;

    void compute_idf_from_profiles(const std::vector<std::string>& text_columns);

    const std::unordered_map<int, UserProfile>* profiles = nullptr;
    const std::unordered_map<int, std::unordered_map<int,float>>* user_feats = nullptr;

    const std::unordered_map<int, std::vector<int>>* adj_list = nullptr;

    std::unordered_map<std::string, std::pair<float,float>> field_normalizers;
    std::unordered_map<std::string, std::pair<float,float>> column_normalizers;

    std::unordered_map<std::string, std::unordered_map<int,float>> idf_per_col;
    size_t total_users = 0;

private:
    std::vector<std::string> text_columns_internal;

    float tfidf_cosine_for_column(const std::unordered_map<int,int>& A,
                                  const std::unordered_map<int,int>& B,
                                  const std::unordered_map<int,float>& idf_map) const;

    static float vec_set_similarity(const std::vector<uint32_t>& A, const std::vector<uint32_t>& B);
    static float region_similarity_local(const std::array<int,3>& A, const std::array<int,3>& B);
    static float cosine_counts_maps_local(const std::unordered_map<int,int>& A, const std::unordered_map<int,int>& B);

    friend struct ::RecommenderInternalGraph;
    friend struct ::RecommenderInternalClubs;
    friend struct ::RecommenderInternalSim;
};

#endif
