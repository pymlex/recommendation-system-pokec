#include "utils.h"
#include "user_profile.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <iostream>

using namespace std;

vector<string> load_text_columns_from_file(const string& path) {
    vector<string> out;
    ifstream in(path);
    if (!in.is_open()) return out;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        out.push_back(line);
    }
    in.close();
    return out;
}

unordered_map<int, vector<int>> build_adj_list(const unordered_map<int, vector<pair<int,float>>>& adj_weighted) {
    unordered_map<int, vector<int>> out;
    for (auto it = adj_weighted.begin(); it != adj_weighted.end(); ++it) {
        int u = it->first;
        const vector<pair<int,float>>& vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i) out[u].push_back(vec[i].first);
    }
    return out;
}

vector<string> split_csv_line(const string& line)
{
    vector<string> out;
    string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        char c = line[i];
        if (c == '"') { in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

vector<pair<int,int>> parse_tok_field(const string& field) {
    vector<pair<int,int>> out;
    if (field.empty()) return out;
    string s = field;
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size()-2);
    stringstream ss(s);
    string tok;
    while (getline(ss, tok, ';')) {
        if (tok.empty()) continue;
        size_t p = tok.find(':');
        if (p == string::npos) continue;
        int id = atoi(tok.substr(0,p).c_str());
        int cnt = atoi(tok.substr(p+1).c_str());
        out.emplace_back(id, cnt);
    }
    return out;
}

static inline uint64_t pair_key_uint64(int a, int b) {
    uint32_t A = (uint32_t)a;
    uint32_t B = (uint32_t)b;
    if (A > B) std::swap(A,B);
    return ( (uint64_t)A << 32 ) | (uint64_t)B;
}

static float vec_set_similarity_local(const std::vector<uint32_t>& A, const std::vector<uint32_t>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    std::unordered_map<uint32_t,int> cnt;
    for (auto v : A) cnt[v] = 1;
    int inter = 0;
    for (auto v : B) if (cnt.find(v) != cnt.end()) ++inter;
    double denom = sqrt((double)A.size()) * sqrt((double)B.size());
    if (denom <= 0.0) return 0.0f;
    return (float)((double)inter / denom);
}

static float region_similarity_local(const std::array<int,3>& A, const std::array<int,3>& B) {
    int a_cnt = 0, b_cnt = 0, matches = 0;
    for (int i = 0; i < 3; ++i) {
        if (A[i] >= 0) ++a_cnt;
        if (B[i] >= 0) ++b_cnt;
        if (A[i] >= 0 && B[i] >= 0 && A[i] == B[i]) ++matches;
    }
    if (a_cnt == 0 || b_cnt == 0) return 0.0f;
    return (float)((double)matches / (sqrt((double)a_cnt) * sqrt((double)b_cnt)));
}

static float cosine_counts_maps_local(const std::unordered_map<int,int>& A, const std::unordered_map<int,int>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    double dot = 0.0;
    double suma2 = 0.0;
    double sumb2 = 0.0;
    for (auto &pa : A) suma2 += (double)pa.second * pa.second;
    for (auto &pb : B) sumb2 += (double)pb.second * pb.second;
    if (suma2 <= 0.0 || sumb2 <= 0.0) return 0.0f;
    if (A.size() < B.size()) {
        for (auto it = A.begin(); it != A.end(); ++it) {
            auto jt = B.find(it->first);
            if (jt != B.end()) dot += (double) it->second * jt->second;
        }
    } else {
        for (auto it = B.begin(); it != B.end(); ++it) {
            auto jt = A.find(it->first);
            if (jt != A.end()) dot += (double) it->second * jt->second;
        }
    }
    double norm = sqrt(suma2) * sqrt(sumb2);
    if (norm <= 0.0) return 0.0f;
    return (float)(dot / norm);
}

bool load_column_normalizers(const string& path, unordered_map<string, pair<float,float>>& out) {
    out.clear();
    ifstream in(path);
    if (!in.is_open()) return false;
    string line;
    if (!getline(in, line)) { in.close(); return false; }
    while (getline(in, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find(',');
        if (p1 == string::npos) continue;
        size_t p2 = line.find(',', p1 + 1);
        string col = line.substr(0, p1);
        if (p2 == string::npos) continue;
        float mean = (float)atof(line.substr(p1 + 1, p2 - (p1 + 1)).c_str());
        float stddev = (float)atof(line.substr(p2 + 1).c_str());
        out[col] = make_pair(mean, stddev);
    }
    in.close();
    return !out.empty();
}

bool save_column_normalizers(const string& path, const unordered_map<string, pair<float,float>>& m) {
    ofstream out(path);
    if (!out.is_open()) return false;
    out << "column,mean,stddev\n";
    for (auto &kv : m) {
        out << kv.first << "," << kv.second.first << "," << kv.second.second << "\n";
    }
    out.close();
    return true;
}

unordered_map<string, pair<float,float>> compute_column_normalizers(
    const unordered_map<int, UserProfile>& profiles,
    const vector<string>& text_columns,
    int sample_size,
    int comps_per_user)
{
    unordered_map<string, pair<float,float>> result;
    if (profiles.empty()) return result;
    vector<int> ids;
    ids.reserve(profiles.size());
    for (auto &kv : profiles) ids.push_back(kv.first);
    mt19937 rng(12345);
    uniform_int_distribution<size_t> dist(0, ids.size() - 1);
    size_t total_needed = (size_t)sample_size * (size_t)comps_per_user;
    unordered_set<uint64_t> seen_pairs;
    vector<vector<double>> vals_text(text_columns.size());
    unordered_map<string, vector<double>> vals_field;
    vector<string> field_keys = {"public","gender","completion","age","region","clubs","friends"};
    for (auto &k : field_keys) vals_field[k] = vector<double>();
    size_t attempts = 0;
    while (seen_pairs.size() < total_needed && attempts < total_needed * 10) {
        ++attempts;
        int a = ids[ dist(rng) ];
        int b = ids[ dist(rng) ];
        if (a == b) continue;
        uint64_t key = pair_key_uint64(a,b);
        if (!seen_pairs.insert(key).second) continue;
        const UserProfile &A = profiles.at(a);
        const UserProfile &B = profiles.at(b);
        double s_pub = 0.0;
        if (A.public_flag >= 0 && B.public_flag >= 0 && A.public_flag == B.public_flag) s_pub = 1.0;
        vals_field["public"].push_back(s_pub);
        double s_gen = 0.0;
        if (A.gender >= 0 && B.gender >= 0 && A.gender == B.gender) s_gen = 1.0;
        vals_field["gender"].push_back(s_gen);
        double s_comp = 0.0;
        if (A.completion_percentage > 0 && B.completion_percentage > 0) {
            int amin = std::min(A.completion_percentage, B.completion_percentage);
            int amax = std::max(A.completion_percentage, B.completion_percentage);
            if (amax > 0) s_comp = (double)amin / (double)amax;
        }
        vals_field["completion"].push_back(s_comp);
        double s_age = 0.0;
        if (A.age > 0 && B.age > 0) {
            int amin = std::min(A.age, B.age);
            int amax = std::max(A.age, B.age);
            if (amax > 0) s_age = (double)amin / (double)amax;
        }
        vals_field["age"].push_back(s_age);
        double s_reg = region_similarity_local(A.region_parts, B.region_parts);
        vals_field["region"].push_back(s_reg);
        double s_clubs = vec_set_similarity_local(A.clubs, B.clubs);
        vals_field["clubs"].push_back(s_clubs);
        double s_friends = vec_set_similarity_local(A.friends, B.friends);
        vals_field["friends"].push_back(s_friends);
        for (size_t t = 0; t < text_columns.size(); ++t) {
            double s = 0.0;
            if (t < A.token_cols.size() && t < B.token_cols.size()) {
                s = cosine_counts_maps_local(A.token_cols[t], B.token_cols[t]);
            }
            vals_text[t].push_back(s);
        }
    }

    auto compute_mean_std = [](const vector<double>& v)->pair<float,float>{
        if (v.empty()) return make_pair(0.0f, 1.0f);
        double mu = 0.0;
        for (double x : v) mu += x;
        mu /= (double)v.size();
        double s = 0.0;
        if (v.size() > 1) {
            for (double x : v) { double d = x - mu; s += d*d; }
            s = sqrt(s / (double)(v.size() - 1));
            if (s == 0.0) s = 1.0;
        } else s = 1.0;
        return make_pair((float)mu, (float)s);
    };

    for (auto &kv : vals_field) {
        result[kv.first] = compute_mean_std(kv.second);
    }
    for (size_t t = 0; t < text_columns.size(); ++t) {
        result[text_columns[t]] = compute_mean_std(vals_text[t]);
    }
    return result;
}
