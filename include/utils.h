#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <unordered_map>

std::vector<std::string> load_text_columns_from_file(const std::string& path);

std::unordered_map<int, std::vector<int>> build_adj_list(const std::unordered_map<int, std::vector<std::pair<int,float>>>& adj_weighted);

bool load_column_normalizers(const std::string& path, std::unordered_map<std::string, std::pair<float,float>>& out);
bool save_column_normalizers(const std::string& path, const std::unordered_map<std::string, std::pair<float,float>>& m);

std::unordered_map<std::string, std::pair<float,float>> compute_column_normalizers(
    const std::unordered_map<int, struct UserProfile>& profiles,
    const std::vector<std::string>& text_columns,
    int sample_size,
    int comps_per_user);

std::vector<std::string> split_csv_line(const std::string& line);
std::vector<std::pair<int,int>> parse_tok_field(const std::string& field);

#endif
