#ifndef UI_H
#define UI_H

#include <unordered_map>
#include <vector>
#include <string>
#include "user_profile.h"
#include "recommender.h"

void run_terminal_ui(std::unordered_map<int, UserProfile>& profiles,
                     const std::unordered_map<int, std::vector<int>>& adj_list,
                     Recommender& rec,
                     const std::unordered_map<int, std::string>& club_id_to_name,
                     const std::vector<std::string>& text_columns,
                     size_t loaded_users);

#endif
