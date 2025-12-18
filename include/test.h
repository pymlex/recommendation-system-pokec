#ifndef FRIENDS_HOLDOUT_TEST_H
#define FRIENDS_HOLDOUT_TEST_H

#include <unordered_map>
#include <vector>
#include <string>

struct UserProfile;
class Recommender;

void run_friends_holdout_test(const std::unordered_map<int, UserProfile>& profiles,
                              const std::unordered_map<int, std::vector<int>>& adj_list,
                              const std::vector<std::string>& text_columns,
                              const Recommender& base_rec,
                              int sample_size,
                              const std::string& out_path);

#endif