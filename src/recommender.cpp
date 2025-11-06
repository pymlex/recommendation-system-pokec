#include "recommender.h"
#include "user_profile.h"

#include <cmath>
#include <algorithm>

using namespace std;

Recommender::Recommender(const unordered_map<int, UserProfile>* profiles_in,
                         const unordered_map<int, vector<int>>* al)
{
    profiles = profiles_in;
    user_feats = nullptr;
    adj_list = al;
    total_users = profiles ? profiles->size() : 0;
}

Recommender::Recommender(const unordered_map<int, unordered_map<int,float>>* user_feats_in,
                         const unordered_map<int, vector<int>>* al)
{
    user_feats = user_feats_in;
    profiles = nullptr;
    adj_list = al;
    total_users = user_feats ? user_feats->size() : 0;
}

void Recommender::set_field_normalizers(const unordered_map<string, pair<float,float>>& m) {
    field_normalizers = m;
}

void Recommender::set_column_normalizers(const unordered_map<string, pair<float,float>>& m) {
    column_normalizers = m;
}

void Recommender::set_text_columns(const vector<string>& cols) {
    text_columns_internal = cols;
}

void Recommender::set_tfidf_index(const unordered_map<string, unordered_map<int,float>>& idf_map) {
    idf_per_col = idf_map;
}

void Recommender::compute_idf_from_profiles(const vector<string>& text_columns)
{
    idf_per_col.clear();
    if (!profiles) return;
    total_users = profiles->size();
    for (size_t t = 0; t < text_columns.size(); ++t) {
        unordered_map<int,int> df;
        for (auto &kv : *profiles) {
            const UserProfile &p = kv.second;
            if (t < p.token_cols.size()) {
                const auto &mp = p.token_cols[t];
                for (auto &pr : mp) {
                    df[pr.first] += 1;
                }
            }
        }
        unordered_map<int,float> idfmap;
        for (auto &pr : df) {
            float idf = logf(1.0f + (float)total_users / (1.0f + (float)pr.second));
            idfmap[pr.first] = idf;
        }
        idf_per_col[text_columns[t]] = std::move(idfmap);
    }
}

float Recommender::tfidf_cosine_for_column(const unordered_map<int,int>& A,
                                           const unordered_map<int,int>& B,
                                           const unordered_map<int,float>& idf_map) const
{
    if (A.empty() || B.empty()) return 0.0f;
    double dot = 0.0;
    double na = 0.0, nb = 0.0;
    if (A.size() < B.size()) {
        for (auto &pa : A) {
            int token = pa.first;
            double idf = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
            double wA = (double)pa.second * idf;
            na += wA * wA;
            auto it = B.find(token);
            if (it != B.end()) {
                double idf_b = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
                double wB = (double)it->second * idf_b;
                dot += wA * wB;
            }
        }
        for (auto &pb : B) {
            int token = pb.first;
            double idf = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
            double wB = (double)pb.second * idf;
            nb += wB * wB;
        }
    } else {
        for (auto &pb : B) {
            int token = pb.first;
            double idf = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
            double wB = (double)pb.second * idf;
            nb += wB * wB;
            auto it = A.find(token);
            if (it != A.end()) {
                double idf_a = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
                double wA = (double)it->second * idf_a;
                dot += wA * wB;
            }
        }
        for (auto &pa : A) {
            int token = pa.first;
            double idf = (idf_map.count(token) ? idf_map.at(token) : 1.0f);
            double wA = (double)pa.second * idf;
            na += wA * wA;
        }
    }
    double denom = sqrt(na) * sqrt(nb);
    if (denom <= 0.0) return 0.0f;
    return (float)(dot / denom);
}

float Recommender::vec_set_similarity(const vector<uint32_t>& A, const vector<uint32_t>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    unordered_map<uint32_t,int> cnt;
    for (auto v : A) cnt[v] = 1;
    int inter = 0;
    for (auto v : B) if (cnt.find(v) != cnt.end()) ++inter;
    double denom = sqrt((double)A.size()) * sqrt((double)B.size());
    if (denom <= 0.0) return 0.0f;
    return (float)((double)inter / denom);
}

float Recommender::region_similarity_local(const array<int,3>& A, const array<int,3>& B) {
    int a_cnt = 0, b_cnt = 0, matches = 0;
    for (int i = 0; i < 3; ++i) {
        if (A[i] >= 0) ++a_cnt;
        if (B[i] >= 0) ++b_cnt;
        if (A[i] >= 0 && B[i] >= 0 && A[i] == B[i]) ++matches;
    }
    if (a_cnt == 0 || b_cnt == 0) return 0.0f;
    return (float)((double)matches / (sqrt((double)a_cnt) * sqrt((double)b_cnt)));
}

float Recommender::cosine_counts_maps_local(const unordered_map<int,int>& A, const unordered_map<int,int>& B) {
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
