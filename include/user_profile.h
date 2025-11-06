#ifndef USER_PROFILE_H
#define USER_PROFILE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <cstdint>

struct UserProfile {
    int user_id = -1;
    int public_flag = -1;
    int completion_percentage = -1;
    int gender = -1;
    int age = 0;
    std::vector<uint32_t> clubs;
    std::vector<uint32_t> friends;
    std::vector< std::unordered_map<int,int> > token_cols;
    std::array<int,3> region_parts = { -1, -1, -1 };
};

#endif
