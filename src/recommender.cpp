#include "recommender.h"
#include "user_profile.h"
#include "tfidf_index.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <unordered_set>

using namespace std;

Recommender::Recommender(unordered_map<int, unordered_map<int,float>>* uf, unordered_map<int, vector<int>>* al)
{
    user_feats = uf;
    adj_list = al;
    profiles = nullptr;
    tfidf = nullptr;
}

Recommender::Recommender(unordered_map<int, UserProfile>* profiles_in, unordered_map<int, vector<int>>* al)
{
    profiles = profiles_in;
    adj_list = al;
    user_feats = nullptr;
    tfidf = nullptr;
}

void Recommender::set_column_normalizers(const vector<pair<float,float>>& norms) {
    column_normalizers = norms;
}
void Recommender::set_field_normalizers(const unordered_map<string, pair<float,float>>& m) {
    field_normalizers = m;
}
void Recommender::set_text_columns(const vector<string>& cols) { text_columns = cols; }
void Recommender::set_tfidf_index(TFIDFIndex* idx) { tfidf = idx; }

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

vector<pair<int,float>> Recommender::recommend_by_profile(int user, int topk)
{
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

float Recommender::profile_similarity(const UserProfile &A, const UserProfile &B, const vector<string> &text_columns) const
{
    int used = 0;
    double sum_z = 0.0;

    double s_pub = 0.0;
    if (A.public_flag >= 0 && B.public_flag >= 0 && A.public_flag == B.public_flag) s_pub = 1.0;
    auto itp = field_normalizers.find("public");
    if (itp != field_normalizers.end()) { sum_z += ((s_pub - itp->second.first) / itp->second.second); ++used; }

    double s_gen = 0.0;
    if (A.gender >= 0 && B.gender >= 0 && A.gender == B.gender) s_gen = 1.0;
    auto itg = field_normalizers.find("gender");
    if (itg != field_normalizers.end()) { sum_z += ((s_gen - itg->second.first) / itg->second.second); ++used; }

    double s_age = 0.0;
    if (A.age > 0 && B.age > 0) {
        int amin = min(A.age, B.age);
        int amax = max(A.age, B.age);
        if (amax > 0) s_age = (double)amin / (double)amax;
    }
    auto ita = field_normalizers.find("age");
    if (ita != field_normalizers.end()) { sum_z += ((s_age - ita->second.first) / ita->second.second); ++used; }

    double s_region = region_similarity_local(A.region_parts, B.region_parts);
    auto itr = field_normalizers.find("region");
    if (itr != field_normalizers.end()) { sum_z += ((s_region - itr->second.first) / itr->second.second); ++used; }

    double s_clubs = vec_set_similarity(A.clubs, B.clubs);
    auto itcl = field_normalizers.find("clubs");
    if (itcl != field_normalizers.end()) { sum_z += ((s_clubs - itcl->second.first) / itcl->second.second); ++used; }

    double s_friends = vec_set_similarity(A.friends, B.friends);
    auto itfr = field_normalizers.find("friends");
    if (itfr != field_normalizers.end()) { sum_z += ((s_friends - itfr->second.first) / itfr->second.second); ++used; }

    size_t T = text_columns.size();
    for (size_t t = 0; t < T; ++t) {
        double s = 0.0;
        if (t < A.token_cols.size() && t < B.token_cols.size()) {
            if (tfidf) s = tfidf->weighted_cosine(A.token_cols[t], B.token_cols[t], (int)t);
            else s = cosine_counts_maps_local(A.token_cols[t], B.token_cols[t]);
        }
        if (t < column_normalizers.size()) {
            auto pr = column_normalizers[t];
            sum_z += ((s - pr.first) / pr.second);
            ++used;
        }
    }

    if (used == 0) return 0.0f;

    double z_avg = sum_z / (double)used;
    double S = 1.0 / (1.0 + exp(-z_avg)); // sigmoid to (0,1)

    double F = 0.0;
    if (A.completion_percentage > 0 && B.completion_percentage > 0) {
        int amin = min(A.completion_percentage, B.completion_percentage);
        int amax = max(A.completion_percentage, B.completion_percentage);
        if (amax > 0) F = (double)amin / (double)amax;
    }

    if (S + F <= 0.0) return 0.0f;
    double FAS = 2.0 * S * F / (S + F);
    return (float)FAS;
}

vector<pair<int,float>> Recommender::recommend_by_cosine(int user, int topk)
{
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

vector<pair<int,float>> Recommender::recommend_from_supernodes(int user, const unordered_map<int, unordered_map<int,float>>& super_feats, int topk)
{
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

// helper gather candidates (friends + friends of friends)
static vector<int> gather_candidates(const unordered_map<int, vector<int>>& adj, int user, int max_candidates) {
    vector<int> out;
    auto it = adj.find(user);
    if (it == adj.end()) return out;
    const vector<int>& fr = it->second;
    unordered_set<int> seen;
    for (int f : fr) { if (f != user) { seen.insert(f); out.push_back(f); } }
    for (int f : fr) {
        auto it2 = adj.find(f);
        if (it2 == adj.end()) continue;
        for (int ff : it2->second) {
            if (ff == user) continue;
            if (seen.insert(ff).second) out.push_back(ff);
            if ((int)out.size() >= max_candidates) return out;
        }
        if ((int)out.size() >= max_candidates) return out;
    }
    return out;
}

vector<pair<int,float>> Recommender::recommend_friends_graph(int user, int topk, int max_candidates)
{
    vector<pair<int,float>> out;
    if (!profiles || !adj_list) return out;
    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;
    const UserProfile &q = itq->second;
    set<int> existing;
    auto itadj = adj_list->find(user);
    if (itadj != adj_list->end()) for (int x : itadj->second) existing.insert(x);
    vector<int> candidates = gather_candidates(*adj_list, user, max_candidates);
    for (int cand : candidates) {
        if (cand == user) continue;
        if (existing.find(cand) != existing.end()) continue;
        auto itc = profiles->find(cand);
        if (itc == profiles->end()) continue;
        float s = profile_similarity(q, itc->second, text_columns);
        out.emplace_back(cand, s);
    }
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_friends_collab(int user, int topk, int max_candidates)
{
    vector<pair<int,float>> out;
    if (!profiles || !adj_list) return out;
    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;
    const UserProfile &q = itq->second;
    set<int> existing;
    auto itadj = adj_list->find(user);
    if (itadj != adj_list->end()) for (int x : itadj->second) existing.insert(x);
    if (itadj == adj_list->end()) return out;

    unordered_map<int,double> score;
    for (int d1 : itadj->second) {
        auto itd1 = profiles->find(d1);
        if (itd1 == profiles->end()) continue;
        double s_u_d1 = profile_similarity(q, itd1->second, text_columns);
        if (s_u_d1 <= 0.0) continue;
        auto itadj2 = adj_list->find(d1);
        if (itadj2 == adj_list->end()) continue;
        for (int d2 : itadj2->second) {
            if (d2 == user) continue;
            if (existing.find(d2) != existing.end()) continue;
            auto itd2 = profiles->find(d2);
            if (itd2 == profiles->end()) continue;
            double s_d1_d2 = profile_similarity(itd1->second, itd2->second, text_columns);
            double add = s_u_d1 * s_d1_d2;
            score[d2] += add;
        }
    }
    for (auto &kv : score) out.emplace_back(kv.first, (float)kv.second);
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_friends_by_interest(int user, int topk, int max_candidates)
{
    vector<pair<int,float>> out;
    if (!profiles || !adj_list) return out;
    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;
    const UserProfile &q = itq->second;
    set<int> existing;
    auto itadj = adj_list->find(user);
    if (itadj != adj_list->end()) for (int x : itadj->second) existing.insert(x);
    vector<int> candidates = gather_candidates(*adj_list, user, max_candidates);
    for (int cand : candidates) {
        if (cand == user) continue;
        if (existing.find(cand) != existing.end()) continue;
        auto itc = profiles->find(cand);
        if (itc == profiles->end()) continue;
        float s = profile_similarity(q, itc->second, text_columns);
        out.emplace_back(cand, s);
    }
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}
