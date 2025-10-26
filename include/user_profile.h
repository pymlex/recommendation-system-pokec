#ifndef USER_PROFILE_H
#define USER_PROFILE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <array>

using namespace std;

struct UserProfile {
    int user_id = -1;
    int public_flag = -1;
    string gender;
    int age = 0;
    vector<int> clubs;
    vector<int> friends;
    vector< unordered_map<int,int> > token_cols;
    array<int,3> region_parts = { -1, -1, -1 };
};

bool load_users_encoded(const string& users_encoded_csv,
                        const vector<string>& text_columns,
                        unordered_map<int, UserProfile>& out_profiles,
                        int& out_median_age);

#endif
