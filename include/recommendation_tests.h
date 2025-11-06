#ifndef RECOMMENDATION_TESTS_H
#define RECOMMENDATION_TESTS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <utility>

struct UserProfile;

struct Recommender; // forward

struct RecommendTestMetrics {
    double graph_hit_rate = 0.0;
    double collab_hit_rate = 0.0;
    double interest_hit_rate = 0.0;
    double avg_club_prec_at_k = 0.0;
    double avg_club_recall_at_k = 0.0;
};

void print_example_recommendations(const std::unordered_map<int, UserProfile>& profiles,
                                   const std::unordered_map<int, std::vector<int>>& adj_list,
                                   Recommender& rec,
                                   const std::unordered_map<int, std::string>& club_id_to_name,
                                   const std::vector<std::string>& text_columns);

RecommendTestMetrics run_recommendation_tests_sample(const std::unordered_map<int, UserProfile>& profiles,
                                                     const std::unordered_map<int, std::vector<int>>& adj_list,
                                                     const std::unordered_map<int, std::string>& club_id_to_name,
                                                     Recommender& base_rec,
                                                     const std::vector<std::string>& text_columns,
                                                     int sample_size = 1000,
                                                     int topk = 10);

#endif
