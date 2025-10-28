#include "utils.h"
#include "user_profile.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>

using namespace std;

unordered_map<int, vector<int>> build_adj_list(const unordered_map<int, vector<pair<int,float>>>& adj_weighted) {
    unordered_map<int, vector<int>> out;
    for (auto it = adj_weighted.begin(); it != adj_weighted.end(); ++it) {
        int u = it->first;
        const vector<pair<int,float>>& vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i) out[u].push_back(vec[i].first);
    }
    return out;
}

static float compute_dot(const unordered_map<int,float>& a, const unordered_map<int,float>& b) {
    float dot = 0.0f;
    if (a.size() < b.size()) {
        for (auto &pa : a) {
            auto it = b.find(pa.first);
            if (it != b.end()) dot += pa.second * it->second;
        }
    } else {
        for (auto &pb : b) {
            auto it = a.find(pb.first);
            if (it != a.end()) dot += pb.second * it->second;
        }
    }
    return dot;
}

float evaluate_holdout_hit_at_k(const unordered_map<int, vector<int>>& adj_list,
                                const unordered_map<int, unordered_map<int,float>>& feats,
                                int sample_size, int k) {
    vector<int> users;
    for (auto it = adj_list.begin(); it != adj_list.end(); ++it) users.push_back(it->first);
    if (users.empty()) return 0.0f;
    random_device rd;
    mt19937 gen(rd());
    shuffle(users.begin(), users.end(), gen);
    int take = sample_size;
    if (take > (int)users.size()) take = (int)users.size();
    int hits = 0;
    for (int i = 0; i < take; ++i) {
        int u = users[i];
        const vector<int>& neigh = adj_list.at(u);
        if (neigh.empty()) continue;
        int held = neigh[0];
        vector<int> reduced;
        for (size_t j = 1; j < neigh.size(); ++j) reduced.push_back(neigh[j]);
        const auto qit = feats.find(u);
        if (qit == feats.end()) continue;
        const unordered_map<int,float>& q = qit->second;
        vector<pair<int,float>> scores;
        for (auto it = feats.begin(); it != feats.end(); ++it) {
            int cand = it->first;
            if (cand == u) continue;
            bool skip = false;
            for (size_t z = 0; z < reduced.size(); ++z) if (reduced[z] == cand) { skip = true; break; }
            if (skip) continue;
            float dot = compute_dot(q, it->second);
            scores.push_back(make_pair(cand, dot));
        }
        sort(scores.begin(), scores.end(), [](const pair<int,float>& A, const pair<int,float>& B){
            if (A.second == B.second) return A.first < B.first;
            return A.second > B.second;
        });
        int limit = k;
        if ((int)scores.size() < limit) limit = (int)scores.size();
        bool hit = false;
        for (int p = 0; p < limit; ++p) if (scores[p].first == held) { hit = true; break; }
        if (hit) ++hits;
    }
    return (float)hits / (float)take;
}

vector<string> load_text_columns_from_file(const string& path) {
    vector<string> out;
    ifstream in(path);
    if (!in.is_open()) return out;
    string line;
    while (getline(in, line)) {
        size_t a = 0;
        while (a < line.size() && isspace((unsigned char)line[a])) ++a;
        size_t b = line.size();
        while (b > a && isspace((unsigned char)line[b-1])) --b;
        if (b <= a) continue;
        string token = line.substr(a, b - a);
        if (token.empty()) continue;
        if (token[0] == '#') continue;
        out.push_back(token);
    }
    return out;
}


static float cosine_counts_maps_local(const unordered_map<int,int>& A, const unordered_map<int,int>& B) {
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

static float vec_set_similarity(const vector<uint32_t>& A, const vector<uint32_t>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    unordered_map<uint32_t,int> cnt;
    cnt.reserve(A.size()*2+1);
    for (auto v : A) cnt[v] = 1;
    int inter = 0;
    for (auto v : B) if (cnt.find(v) != cnt.end()) ++inter;
    double denom = sqrt((double)A.size()) * sqrt((double)B.size());
    if (denom <= 0.0) return 0.0f;
    return (float)((double)inter / denom);
}

static float region_similarity_local(const array<int,3>& A, const array<int,3>& B) {
    int a_cnt = 0, b_cnt = 0, matches = 0;
    for (int i = 0; i < 3; ++i) {
        if (A[i] >= 0) ++a_cnt;
        if (B[i] >= 0) ++b_cnt;
        if (A[i] >= 0 && B[i] >= 0 && A[i] == B[i]) ++matches;
    }
    if (a_cnt == 0 || b_cnt == 0) return 0.0f;
    return (float)((double)matches / (sqrt((double)a_cnt) * sqrt((double)b_cnt)));
}

bool load_median_age(const string& path, int& out_median) {
    out_median = 0;
    ifstream in(path);
    if (!in.is_open()) return false;
    string line;
    if (!getline(in, line)) { in.close(); return false; }
    out_median = atoi(line.c_str());
    in.close();
    if (out_median <= 0) return false;
    return true;
}

bool save_median_age(const string& path, int median) {
    ofstream out(path);
    if (!out.is_open()) return false;
    out << median << "\n";
    out.close();
    return true;
}

int compute_median_age_from_profiles(const unordered_map<int, UserProfile>& profiles) {
    vector<int> ages;
    ages.reserve(profiles.size());
    for (auto &kv : profiles) {
        int a = kv.second.age;
        if (a > 0) ages.push_back(a);
    }
    if (ages.empty()) return 0;
    sort(ages.begin(), ages.end());
    size_t n = ages.size();
    if (n % 2) return ages[n/2];
    return (ages[n/2 - 1] + ages[n/2]) / 2;
}

int fill_missing_ages(unordered_map<int, UserProfile>& profiles, int median_age) {
    int replaced = 0;
    for (auto &kv : profiles) {
        if (kv.second.age == 0) { kv.second.age = median_age; ++replaced; }
    }
    return replaced;
}

unordered_map<string, pair<float,float>> compute_column_normalizers(
    const unordered_map<int, UserProfile>& profiles,
    const vector<string>& text_columns,
    size_t sample_users,
    int comps_per_user
) {
    unordered_map<string, pair<float,float>> res;

    vector<int> uids;
    uids.reserve(profiles.size());
    for (auto &kv : profiles) uids.push_back(kv.first);
    size_t N = uids.size();
    if (N == 0) return res;

    size_t sample_n = sample_users < N ? sample_users : N;

    std::mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    unordered_set<uint64_t> seen_pairs;
    vector<int> sample_ids;
    sample_ids.reserve(sample_n);

    for (size_t i = 0; i < sample_n; ++i) {
        std::uniform_int_distribution<size_t> dist(0, N-1);
        size_t idx = dist(rng);
        sample_ids.push_back(uids[idx]);
    }

    vector<string> field_names;
    field_names.push_back("public");
    field_names.push_back("gender");
    field_names.push_back("completion");
    field_names.push_back("age");
    field_names.push_back("region");
    field_names.push_back("clubs");
    field_names.push_back("friends");
    for (auto &c : text_columns) field_names.push_back(c);

    unordered_map<string, vector<double>> accum;
    for (auto &fn : field_names) accum[fn] = vector<double>();

    for (size_t si = 0; si < sample_ids.size(); ++si) {
        int a_id = sample_ids[si];
        for (int k = 0; k < comps_per_user; ++k) {
            std::uniform_int_distribution<size_t> dist2(0, N-1);
            int b_id = uids[dist2(rng)];
            if (b_id == a_id) continue;
            uint64_t key = ((uint64_t)std::min(a_id,b_id) << 32) | (uint64_t)std::max(a_id,b_id);
            if (seen_pairs.find(key) != seen_pairs.end()) continue;
            seen_pairs.insert(key);

            const UserProfile &A = profiles.at(a_id);
            const UserProfile &B = profiles.at(b_id);

            double s_public = 0.0;
            if (A.public_flag >= 0 && B.public_flag >= 0 && A.public_flag == B.public_flag) s_public = 1.0;
            accum["public"].push_back(s_public);

            double s_gender = 0.0;
            if (A.gender >= 0 && B.gender >= 0 && A.gender == B.gender) s_gender = 1.0;
            accum["gender"].push_back(s_gender);

            double s_completion = 0.0;
            if (A.completion_percentage >= 0 && B.completion_percentage >= 0) {
                int amin = std::min(A.completion_percentage, B.completion_percentage);
                int amax = std::max(A.completion_percentage, B.completion_percentage);
                if (amax > 0) s_completion = (double)amin / (double)amax;
            }
            accum["completion"].push_back(s_completion);

            double s_age = 0.0;
            if (A.age > 0 && B.age > 0) {
                int amin = std::min(A.age, B.age);
                int amax = std::max(A.age, B.age);
                if (amax > 0) s_age = (double)amin / (double)amax;
            }
            accum["age"].push_back(s_age);

            double s_region = region_similarity_local(A.region_parts, B.region_parts);
            accum["region"].push_back(s_region);

            double s_clubs = vec_set_similarity(A.clubs, B.clubs);
            accum["clubs"].push_back(s_clubs);

            double s_friends = vec_set_similarity(A.friends, B.friends);
            accum["friends"].push_back(s_friends);

            for (size_t t = 0; t < text_columns.size(); ++t) {
                double s_text = 0.0;
                if (t < A.token_cols.size() && t < B.token_cols.size()) {
                    s_text = cosine_counts_maps_local(A.token_cols[t], B.token_cols[t]);
                }
                accum[text_columns[t]].push_back(s_text);
            }
        }
    }

    for (auto &kv : accum) {
        const string &name = kv.first;
        vector<double> &vals = kv.second;
        if (vals.empty()) {
            res[name] = make_pair(0.0f, 0.0f);
            continue;
        }
        double mean = 0.0;
        for (double v : vals) mean += v;
        mean /= (double)vals.size();
        double var = 0.0;
        if (vals.size() > 1) {
            for (double v : vals) { double d = v - mean; var += d * d; }
            var /= (double)(vals.size() - 1);
        } else var = 0.0;
        double stddev = sqrt(var);
        res[name] = make_pair((float)mean, (float)stddev);
    }

    return res;
}

bool load_column_normalizers(const string& path, unordered_map<string, pair<float,float>>& out) {
    out.clear();
    ifstream in(path);
    if (!in.is_open()) return false;
    string line;
    if (!getline(in, line)) { in.close(); return false; }
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string field;
        string mean_s, std_s;
        if (!getline(ss, field, ',')) continue;
        if (!getline(ss, mean_s, ',')) continue;
        if (!getline(ss, std_s, ',')) continue;
        float mean = (float)atof(mean_s.c_str());
        float stddev = (float)atof(std_s.c_str());
        out[field] = make_pair(mean, stddev);
    }
    in.close();
    return !out.empty();
}

bool save_column_normalizers(const string& path, const unordered_map<string, pair<float,float>>& in_map) {
    ofstream out(path);
    if (!out.is_open()) return false;
    out << "field,mean,stddev\n";
    for (auto &kv : in_map) {
        out << kv.first << "," << kv.second.first << "," << kv.second.second << "\n";
    }
    out.close();
    return true;
}