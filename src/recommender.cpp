#include "recommender.h"
#include "user_profile.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <unordered_set>

using namespace std;

static float cosine_map_counts(const unordered_map<int,int>& A, const unordered_map<int,int>& B)
{
    if (A.empty() && B.empty()) return 0.0f;
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (auto &p : A) normA += (double)p.second * (double)p.second;
    for (auto &p : B) normB += (double)p.second * (double)p.second;
    if (normA <= 0.0 || normB <= 0.0) return 0.0f;
    for (auto &p : A) {
        auto it = B.find(p.first);
        if (it != B.end()) dot += (double)p.second * (double)it->second;
    }
    double denom = sqrt(normA) * sqrt(normB);
    if (denom <= 0.0) return 0.0f;
    return (float)(dot / denom);
}

static float jaccard_sets(const vector<int>& A, const vector<int>& B)
{
    if (A.empty() && B.empty()) return 0.0f;
    unordered_set<int> sA;
    for (int x : A) sA.insert(x);
    unordered_set<int> sB;
    for (int x : B) sB.insert(x);
    int inter = 0;
    for (int x : sA) if (sB.find(x) != sB.end()) ++inter;
    int uni = (int)(sA.size() + sB.size() - inter);
    if (uni == 0) return 0.0f;
    return (float)inter / (float)uni;
}

Recommender::Recommender(unordered_map<int, unordered_map<int,float>>* uf,
                         unordered_map<int, vector<int>>* al)
    : user_feats(uf), profiles(nullptr), adj_list(al)
{
}

Recommender::Recommender(unordered_map<int, UserProfile>* profiles_in,
                         unordered_map<int, vector<int>>* al)
    : user_feats(nullptr), profiles(profiles_in), adj_list(al)
{
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

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B) {
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });

    if ((int) out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_from_supernodes(int user,
    const unordered_map<int, unordered_map<int,float>>& super_feats, int topk)
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

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B) {
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });

    if ((int) out.size() > topk) out.resize(topk);
    return out;
}

float Recommender::profile_similarity(const UserProfile& a, const UserProfile& b) const
{
    double sum = 0.0;
    int count = 0;

    if (!a.gender.empty() || !b.gender.empty()) {
        if (!a.gender.empty() && !b.gender.empty()) sum += (a.gender == b.gender ? 1.0 : 0.0);
        else sum += 0.0;
        ++count;
    }

    if (a.public_flag != -1 || b.public_flag != -1) {
        if (a.public_flag != -1 && b.public_flag != -1) sum += (a.public_flag == b.public_flag ? 1.0 : 0.0);
        else sum += 0.0;
        ++count;
    }

    if (a.region_id != -1 || b.region_id != -1) {
        if (a.region_id != -1 && b.region_id != -1) sum += (a.region_id == b.region_id ? 1.0 : 0.0);
        else sum += 0.0;
        ++count;
    }

    if (a.age > 0 || b.age > 0) {
        if (a.age > 0 && b.age > 0) {
            int mn = std::min(a.age, b.age);
            int mx = std::max(a.age, b.age);
            if (mx > 0) sum += (double)mn / (double)mx; else sum += 0.0;
        } else sum += 0.0;
        ++count;
    }

    if (!a.clubs.empty() || !b.clubs.empty()) {
        sum += jaccard_sets(a.clubs, b.clubs);
        ++count;
    }

    size_t tcols = min(a.token_cols.size(), b.token_cols.size());
    for (size_t i = 0; i < tcols; ++i) {
        const unordered_map<int,int>& A = a.token_cols[i];
        const unordered_map<int,int>& B = b.token_cols[i];
        if (A.empty() && B.empty()) continue;
        sum += cosine_map_counts(A, B);
        ++count;
    }

    if (count == 0) return 0.0f;
    double res = sum / (double) count;
    if (res < 0.0) res = 0.0;
    if (res > 1.0) res = 1.0;
    return (float) res;
}

vector<pair<int,float>> Recommender::recommend_by_profile(int user, int topk)
{
    vector<pair<int,float>> out;
    if (!profiles) return out;
    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;

    const UserProfile& q = itq->second;
    set<int> existing;
    if (adj_list && adj_list->find(user) != adj_list->end()) {
        const vector<int>& ne = adj_list->at(user);
        for (size_t i = 0; i < ne.size(); ++i) existing.insert(ne[i]);
    }

    for (auto it = profiles->begin(); it != profiles->end(); ++it) {
        int uid = it->first;
        if (uid == user) continue;
        if (existing.find(uid) != existing.end()) continue;
        float sim = profile_similarity(q, it->second);
        out.push_back(make_pair(uid, sim));
    }

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B) {
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });

    if ((int) out.size() > topk) out.resize(topk);
    return out;
}
