#include "recommender.h"
#include "user_profile.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>

using namespace std;

vector<pair<int,float>> Recommender::recommend_clubs_collab(int user, int topk, int candidate_limit) const
{
    vector<pair<int,float>> out;
    if (!profiles || !adj_list) return out;

    auto itq = profiles->find(user);
    if (itq == profiles->end()) return out;
    const UserProfile &q = itq->second;

    vector<int> friends;
    auto itf = adj_list->find(user);
    if (itf != adj_list->end()) friends = itf->second;

    unordered_map<int,float> sim_u_f;
    for (int f : friends) {
        auto itpf = profiles->find(f);
        if (itpf == profiles->end()) continue;
        sim_u_f[f] = profile_similarity(q, itpf->second);
    }

    unordered_map<int,double> club_scores;
    unordered_set<int> user_clubs;
    for (auto c : q.clubs) user_clubs.insert((int)c);

    for (int f : friends) {
        auto itpf = profiles->find(f);
        if (itpf == profiles->end()) continue;
        double w = (sim_u_f.count(f) ? sim_u_f.at(f) : 0.0);
        if (w <= 0.0) continue;
        for (auto cid : itpf->second.clubs) {
            if (user_clubs.find((int)cid) != user_clubs.end()) continue;
            club_scores[(int)cid] += w;
        }
    }

    for (int f : friends) {
        auto itff = adj_list->find(f);
        if (itff == adj_list->end()) continue;
        auto itpf = profiles->find(f);
        if (itpf == profiles->end()) continue;
        double wuf = (sim_u_f.count(f) ? sim_u_f.at(f) : 0.0);
        if (wuf <= 0.0) continue;
        for (int fof : itff->second) {
            if (fof == user) continue;
            auto itpfof = profiles->find(fof);
            if (itpfof == profiles->end()) continue;
            double s_f_fof = profile_similarity(itpf->second, itpfof->second);
            if (s_f_fof <= 0.0) continue;
            double contrib = wuf * s_f_fof;
            for (auto cid : itpfof->second.clubs) {
                if (user_clubs.find((int)cid) != user_clubs.end()) continue;
                club_scores[(int)cid] += contrib;
            }
        }
    }

    for (auto &kv : club_scores) out.emplace_back(kv.first, (float)kv.second);
    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_from_supernodes(int user, const unordered_map<int, unordered_map<int,float>>& super_feats, int topk) const
{
    vector<pair<int,float>> out;
    if (!adj_list) return out;

    if (user_feats) {
        auto itq = user_feats->find(user);
        if (itq == user_feats->end()) return out;
        const auto &q = itq->second;
        for (auto it = super_feats.begin(); it != super_feats.end(); ++it) {
            int sid = it->first;
            const auto &vec = it->second;
            double dot = 0.0;
            if (!q.empty() && !vec.empty()) {
                if (q.size() < vec.size()) {
                    for (auto &qa : q) {
                        auto jt = vec.find(qa.first);
                        if (jt != vec.end()) dot += (double)qa.second * (double)jt->second;
                    }
                } else {
                    for (auto &pb : vec) {
                        auto at = q.find(pb.first);
                        if (at != q.end()) dot += (double)pb.second * (double)at->second;
                    }
                }
            }
            out.emplace_back(sid, (float)dot);
        }
    } else {
        if (!profiles) return out;
        auto itq = profiles->find(user);
        if (itq == profiles->end()) return out;
        unordered_map<int,float> qvec;
        if (!idf_per_col.empty()) {
            for (auto &kv : idf_per_col) {
                const string &colname = kv.first;
                const auto &idfmap = kv.second;
                int col_idx = -1;
                for (size_t i = 0; i < text_columns_internal.size(); ++i) if (text_columns_internal[i] == colname) { col_idx = (int)i; break; }
                if (col_idx < 0) continue;
                const UserProfile &p = itq->second;
                if (col_idx >= (int)p.token_cols.size()) continue;
                for (auto &pr : p.token_cols[col_idx]) {
                    int token = pr.first;
                    float tf = (float)pr.second;
                    float idf = (idfmap.count(token) ? idfmap.at(token) : 1.0f);
                    qvec[token] += tf * idf;
                }
            }
        }
        for (auto it = super_feats.begin(); it != super_feats.end(); ++it) {
            int sid = it->first;
            const auto &vec = it->second;
            double dot = 0.0;
            if (!qvec.empty() && !vec.empty()) {
                if (qvec.size() < vec.size()) {
                    for (auto &qa : qvec) {
                        auto jt = vec.find(qa.first);
                        if (jt != vec.end()) dot += (double)qa.second * (double)jt->second;
                    }
                } else {
                    for (auto &pb : vec) {
                        auto at = qvec.find(pb.first);
                        if (at != qvec.end()) dot += (double)pb.second * (double)at->second;
                    }
                }
            }
            out.emplace_back(sid, (float)dot);
        }
    }

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){ if (A.second == B.second) return A.first < B.first; return A.second > B.second; });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}
