#include "recommender.h"
#include "user_profile.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>

using namespace std;

Recommender::Recommender(unordered_map<int, unordered_map<int,float>>* uf, unordered_map<int, vector<int>>* al) {
    user_feats = uf;
    adj_list = al;
    profiles = nullptr;
}

Recommender::Recommender(unordered_map<int, UserProfile>* profiles_in, unordered_map<int, vector<int>>* al) {
    profiles = profiles_in;
    adj_list = al;
    user_feats = nullptr;
}

vector<pair<int,float>> Recommender::recommend_by_cosine(int user, int topk) {
    vector<pair<int,float>> out;
    if (!user_feats) return out;
    if (user_feats->find(user) == user_feats->end()) return out;
    const unordered_map<int,float>& q = user_feats->at(user);
    set<int> existing;
    if (adj_list && adj_list->find(user) != adj_list->end()) {
        const vector<int>& ne = adj_list->at(user);
        for (size_t i = 0; i < ne.size(); ++i) existing.insert(ne[i]);
    }
    for (auto it = user_feats->begin(); it != user_feats->end(); ++it) {
        int uid = it->first;
        if (uid == user) continue;
        if (existing.find(uid) != existing.end()) continue;
        const unordered_map<int,float>& v = it->second;
        float dot = 0.0f;
        if (q.size() < v.size()) {
            for (auto ait = q.begin(); ait != q.end(); ++ait) {
                int k = ait->first;
                float va = ait->second;
                auto jt = v.find(k);
                if (jt != v.end()) dot += va * jt->second;
            }
        } else {
            for (auto bit = v.begin(); bit != v.end(); ++bit) {
                int k = bit->first;
                float vb = bit->second;
                auto at = q.find(k);
                if (at != q.end()) dot += vb * at->second;
            }
        }
        out.push_back(make_pair(uid, dot));
    }
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int) out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_from_supernodes(int user, const unordered_map<int, unordered_map<int,float>>& super_feats, int topk) {
    vector<pair<int,float>> out;
    if (!user_feats) return out;
    if (user_feats->find(user) == user_feats->end()) return out;
    const unordered_map<int,float>& q = user_feats->at(user);
    for (auto it = super_feats.begin(); it != super_feats.end(); ++it) {
        int sid = it->first;
        const unordered_map<int,float>& v = it->second;
        float dot = 0.0f;
        if (q.size() < v.size()) {
            for (auto ait = q.begin(); ait != q.end(); ++ait) {
                int k = ait->first;
                float va = ait->second;
                auto jt = v.find(k);
                if (jt != v.end()) dot += va * jt->second;
            }
        } else {
            for (auto bit = v.begin(); bit != v.end(); ++bit) {
                int k = bit->first;
                float vb = bit->second;
                auto at = q.find(k);
                if (at != q.end()) dot += vb * at->second;
            }
        }
        out.push_back(make_pair(sid, dot));
    }
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int) out.size() > topk) out.resize(topk);
    return out;
}

// ---------- profile-based methods ----------
static float cosine_counts_maps_static(const unordered_map<int,int>& A, const unordered_map<int,int>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    double dot = 0.0;
    double suma2 = 0.0;
    double sumb2 = 0.0;
    for (auto &pa : A) suma2 += (double)pa.second * (double)pa.second;
    for (auto &pb : B) sumb2 += (double)pb.second * (double)pb.second;
    if (suma2 <= 0.0 || sumb2 <= 0.0) return 0.0f;
    if (A.size() < B.size()) {
        for (auto it = A.begin(); it != A.end(); ++it) {
            int k = it->first;
            int va = it->second;
            auto jt = B.find(k);
            if (jt != B.end()) dot += (double)va * (double)(jt->second);
        }
    } else {
        for (auto it = B.begin(); it != B.end(); ++it) {
            int k = it->first;
            int vb = it->second;
            auto jt = A.find(k);
            if (jt != A.end()) dot += (double)vb * (double)(jt->second);
        }
    }
    double norm = sqrt(suma2) * sqrt(sumb2);
    if (norm <= 0.0) return 0.0f;
    return (float)(dot / norm);
}

float Recommender::profile_similarity(const UserProfile &A, const UserProfile &B, const vector<string> &text_columns) const {
    int cols_with_value = 0;
    double sum_sim = 0.0;
    bool a_pub_valid = (A.public_flag >= 0);
    bool b_pub_valid = (B.public_flag >= 0);
    if (a_pub_valid || b_pub_valid) {
        ++cols_with_value;
        double s = 0.0;
        if (a_pub_valid && b_pub_valid && A.public_flag == B.public_flag) s = 1.0;
        sum_sim += s;
    }
    bool a_gender = !A.gender.empty();
    bool b_gender = !B.gender.empty();
    if (a_gender || b_gender) {
        ++cols_with_value;
        double s = 0.0;
        if (a_gender && b_gender && A.gender == B.gender) s = 1.0;
        sum_sim += s;
    }
    bool a_age = (A.age > 0);
    bool b_age = (B.age > 0);
    if (a_age || b_age) {
        ++cols_with_value;
        double s = 0.0;
        if (a_age && b_age) {
            int amin = min(A.age, B.age);
            int amax = max(A.age, B.age);
            if (amax > 0) s = (double)amin / (double)amax;
        }
        sum_sim += s;
    }
    bool a_region_nonempty = false, b_region_nonempty = false;
    int a_cnt = 0, b_cnt = 0, matches = 0;
    for (int i = 0; i < 3; ++i) {
        if (A.region_parts[i] >= 0) { a_region_nonempty = true; ++a_cnt; }
        if (B.region_parts[i] >= 0) { b_region_nonempty = true; ++b_cnt; }
        if (A.region_parts[i] >= 0 && B.region_parts[i] >= 0 && A.region_parts[i] == B.region_parts[i]) ++matches;
    }
    if (a_region_nonempty || b_region_nonempty) {
        ++cols_with_value;
        double s = 0.0;
        if (a_cnt > 0 && b_cnt > 0) s = (double)matches / (sqrt((double)a_cnt) * sqrt((double)b_cnt));
        sum_sim += s;
    }

    size_t T = text_columns.size();
    for (size_t t = 0; t < T; ++t) {
        const unordered_map<int,int> *pa = nullptr, *pb = nullptr;
        if (t < A.token_cols.size()) pa = &A.token_cols[t];
        if (t < B.token_cols.size()) pb = &B.token_cols[t];
        bool a_nonempty = (pa && !pa->empty());
        bool b_nonempty = (pb && !pb->empty());
        if (a_nonempty || b_nonempty) {
            ++cols_with_value;
            double s = 0.0;
            if (a_nonempty && b_nonempty) s = cosine_counts_maps_static(*pa, *pb);
            double s_norm = s;
            if (t < column_normalizers.size() && column_normalizers[t] > 0.0f) s_norm = s / (double)column_normalizers[t];
            sum_sim += s_norm;
        }
    }

    if (cols_with_value == 0) return 0.0f;
    return (float)(sum_sim / (double)cols_with_value);
}

vector<pair<int,float>> Recommender::recommend_by_profile(int user, int topk) {
    vector<pair<int,float>> out;
    if (!profiles) return out;
    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;
    const UserProfile &q = itq->second;
    set<int> existing;
    if (adj_list && adj_list->find(user) != adj_list->end()) {
        const vector<int>& ne = adj_list->at(user);
        for (size_t i = 0; i < ne.size(); ++i) existing.insert(ne[i]);
    }
    vector<string> fake_text_cols;
    size_t tcnt = q.token_cols.size();
    fake_text_cols.resize(tcnt);
    for (size_t i = 0; i < tcnt; ++i) fake_text_cols[i] = "col" + to_string(i);
    for (auto it = profiles->begin(); it != profiles->end(); ++it) {
        int uid = it->first;
        if (uid == user) continue;
        if (existing.find(uid) != existing.end()) continue;
        const UserProfile &other = it->second;
        float s = profile_similarity(q, other, fake_text_cols);
        out.push_back(make_pair(uid, s));
    }
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int) out.size() > topk) out.resize(topk);
    return out;
}

void Recommender::set_column_normalizers(const vector<float>& norms) {
    column_normalizers = norms;
    cerr << "[Recommender] column_normalizers set, count=" << norms.size() << "\n";
}
