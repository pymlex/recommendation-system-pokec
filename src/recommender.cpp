#include "recommender.h"
#include <algorithm>
#include <set>
using namespace std;

Recommender::Recommender(unordered_map<int, unordered_map<int,float>>* uf, unordered_map<int, vector<int>>* al)
{
    user_feats = uf;
    adj_list = al;
}

vector<pair<int,float>> Recommender::recommend_by_cosine(int user, int topk)
{
    vector<pair<int,float>> out;
    if (user_feats->find(user) == user_feats->end()) return out;
    const unordered_map<int,float>& q = user_feats->at(user);
    set<int> existing;
    if (adj_list->find(user) != adj_list->end())
    {
        const vector<int>& ne = adj_list->at(user);
        for (size_t i = 0; i < ne.size(); ++i) existing.insert(ne[i]);
    }
    for (auto it = user_feats->begin(); it != user_feats->end(); ++it)
    {
        int uid = it->first;
        if (uid == user) continue;
        if (existing.find(uid) != existing.end()) continue;
        const unordered_map<int,float>& v = it->second;
        float dot = 0.0f;
        if (q.size() < v.size())
        {
            for (auto ait = q.begin(); ait != q.end(); ++ait)
            {
                int k = ait->first;
                float va = ait->second;
                auto jt = v.find(k);
                if (jt != v.end()) dot += va * jt->second;
            }
        }
        else
        {
            for (auto bit = v.begin(); bit != v.end(); ++bit)
            {
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
    if (user_feats->find(user) == user_feats->end()) return out;
    const unordered_map<int,float>& q = user_feats->at(user);
    for (auto it = super_feats.begin(); it != super_feats.end(); ++it)
    {
        int sid = it->first;
        const unordered_map<int,float>& v = it->second;
        float dot = 0.0f;
        if (q.size() < v.size())
        {
            for (auto ait = q.begin(); ait != q.end(); ++ait)
            {
                int k = ait->first;
                float va = ait->second;
                auto jt = v.find(k);
                if (jt != v.end()) dot += va * jt->second;
            }
        }
        else
        {
            for (auto bit = v.begin(); bit != v.end(); ++bit)
            {
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
