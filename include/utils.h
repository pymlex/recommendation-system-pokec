#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <utility>


using namespace std;


struct UserProfile;


unordered_map<int, vector<int>> build_adj_list(const unordered_map<int, vector<pair<int,float>>>& adj_weighted);
float evaluate_holdout_hit_at_k(const unordered_map<int, vector<int>>& adj_list,
                                const unordered_map<int, unordered_map<int,float>>& feats,
                                int sample_size, int k);
vector<string> load_text_columns_from_file(const string& path);

bool load_median_age(const string& path, int& out_median);
bool save_median_age(const string& path, int median);
int compute_median_age_from_profiles(const unordered_map<int, UserProfile>& profiles);
int fill_missing_ages(unordered_map<int, UserProfile>& profiles, int median_age);

unordered_map<string, pair<float,float>> compute_column_normalizers(
    const unordered_map<int, UserProfile>& profiles,
    const vector<string>& text_columns,
    size_t sample_users,
    int comps_per_user
);

bool load_column_normalizers(const string& path, unordered_map<string, pair<float,float>>& out);
bool save_column_normalizers(const string& path, const unordered_map<string, pair<float,float>>& in);

#endif
