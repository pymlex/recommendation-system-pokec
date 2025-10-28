#ifndef COLUMN_STATS_H
#define COLUMN_STATS_H

#include "user_profile.h"
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

unordered_map<string, pair<float,float>> compute_column_mean_similarities(
    const unordered_map<int, UserProfile>& profiles_map,
    const vector<string>& text_columns,
    int sample_size,
    int comps_per_user
);

#endif
