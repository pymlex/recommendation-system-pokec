#ifndef COLUMN_STATS_H
#define COLUMN_STATS_H

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

using namespace std;

struct UserProfile;

vector<float> compute_column_mean_similarities(
    const unordered_map<int, UserProfile>& profiles_map,
    const vector<string>& text_columns,
    int sample_size,
    int comps_per_user
);

bool load_column_normalizers_csv(const string& path,
                                 const vector<string>& text_columns,
                                 vector<float>& out_norms);

#endif
