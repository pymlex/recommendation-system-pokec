#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <unordered_map>
#include <vector>
#include <string>
#include "user_profile.h"

struct EvalMetrics { double graph_hit=0.0; double collab_hit=0.0; double interest_hit=0.0; double supernode_hit=0.0; };

EvalMetrics evaluate_recommenders_holdout(const std::unordered_map<int, UserProfile>& profiles,
                                         const std::unordered_map<int, std::vector<int>>& adj_list,
                                         const std::vector<std::string>& text_columns,
                                         int sample_size,
                                         int topk,
                                         const std::unordered_map<int, std::unordered_map<int,float>>* super_feats = nullptr);

#endif
