#ifndef EVAL_H
#define EVAL_H

#include <unordered_map>
#include "user_profile.h"
#include <vector>

struct EvalResult {
    double hit_at_k;
    double precision_at_k;
    double recall_at_k;
};

EvalResult evaluate_recommender_sample(
    const std::unordered_map<int, UserProfile>& profiles,
    const std::unordered_map<int, std::vector<int>>& adj_list,
    class Recommender &rec,
    const std::vector<std::string>& text_columns,
    int sample_size,
    int k);
#endif
