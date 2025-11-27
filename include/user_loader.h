#ifndef USER_LOADER_H
#define USER_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "user_profile.h"

bool load_users_encoded(const std::string& users_encoded_csv,
                        const std::vector<std::string>& text_columns,
                        std::unordered_map<int, UserProfile>& out_profiles,
                        size_t max_users);

int compute_median_age_from_profiles(const std::unordered_map<int, UserProfile>& profiles);
bool load_median_age(const std::string& path, int& out_median);
bool save_median_age(const std::string& path, int median);
int fill_missing_ages(std::unordered_map<int, UserProfile>& profiles, int median_age);

#endif
