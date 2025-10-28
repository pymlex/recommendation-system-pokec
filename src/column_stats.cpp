#include "column_stats.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <unordered_set>

using namespace std;

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

unordered_map<string, pair<float,float>> compute_column_mean_similarities(
    const unordered_map<int, UserProfile>& profiles_map,
    const vector<string>& text_columns,
    int sample_size,
    int comps_per_user
) {
    unordered_map<string, pair<float,float>> out;
    vector<int> uids;
    uids.reserve(profiles_map.size());
    for (auto &pr : profiles_map) uids.push_back(pr.first);
    if (uids.size() < 2) return out;
    random_device rd;
    mt19937 gen(rd());
    shuffle(uids.begin(), uids.end(), gen);
    size_t n = uids.size();
    unordered_set<uint64_t> pairs_seen;
    unordered_map<string, vector<double>> vals;
    int take = (sample_size > 0 && (size_t)sample_size < n) ? sample_size : (int)n;
    uniform_int_distribution<size_t> dist(0, n - 1);
    for (int i = 0; i < take; ++i) {
        int a = uids[i];
        for (int k = 0; k < comps_per_user; ++k) {
            size_t j = dist(gen);
            int b = uids[j];
            if (a == b) continue;
            uint32_t x = (uint32_t) min(a,b);
            uint32_t y = (uint32_t) max(a,b);
            uint64_t key = ((uint64_t)x << 32) | (uint64_t)y;
            if (pairs_seen.find(key) != pairs_seen.end()) continue;
            pairs_seen.insert(key);
            const UserProfile& A = profiles_map.at(a);
            const UserProfile& B = profiles_map.at(b);
            float pub = 0.0f;
            if (A.public_flag >= 0 && B.public_flag >= 0 && A.public_flag == B.public_flag) pub = 1.0f;
            vals["public"].push_back(pub);
            float gen = 0.0f;
            if (A.gender >= 0 && B.gender >= 0 && A.gender == B.gender) gen = 1.0f;
            vals["gender"].push_back(gen);
            float comp = 0.0f;
            if (A.completion_percentage > 0 && B.completion_percentage > 0) {
                int amin = min(A.completion_percentage, B.completion_percentage);
                int amax = max(A.completion_percentage, B.completion_percentage);
                if (amax > 0) comp = (float)amin / (float)amax;
            }
            vals["completion"].push_back(comp);
            float ag = 0.0f;
            if (A.age > 0 && B.age > 0) {
                int amin = min(A.age, B.age);
                int amax = max(A.age, B.age);
                if (amax > 0) ag = (float)amin / (float)amax;
            }
            vals["age"].push_back(ag);
            float rsim = region_similarity_local(A.region_parts, B.region_parts);
            vals["region"].push_back(rsim);
            float clubsim = vec_set_similarity(A.clubs, B.clubs);
            vals["clubs"].push_back(clubsim);
            float friendsim = vec_set_similarity(A.friends, B.friends);
            vals["friends"].push_back(friendsim);
            size_t T = text_columns.size();
            for (size_t t = 0; t < T; ++t) {
                const unordered_map<int,int> *pa = (t < A.token_cols.size()) ? &A.token_cols[t] : nullptr;
                const unordered_map<int,int> *pb = (t < B.token_cols.size()) ? &B.token_cols[t] : nullptr;
                float s = 0.0f;
                if (pa && pb) s = cosine_counts_maps_local(*pa, *pb);
                string key = text_columns[t];
                vals[key].push_back(s);
            }
        }
    }
    for (auto &kv : vals) {
        const vector<double>& v = kv.second;
        if (v.empty()) continue;
        double sum = 0.0;
        for (double x : v) sum += x;
        double mean = sum / (double)v.size();
        double s2 = 0.0;
        for (double x : v) {
            double d = x - mean;
            s2 += d * d;
        }
        double sd = 0.0;
        if (v.size() > 1) sd = sqrt(s2 / (double)(v.size() - 1));
        out[kv.first] = make_pair((float)mean, (float)sd);
    }
    return out;
}
