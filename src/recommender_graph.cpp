#include "recommender.h"
#include "user_profile.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>

using namespace std;

static void gather_candidates_local(const unordered_map<int, vector<int>>* adj_list, int user, vector<int>& out, int candidate_limit) {
    out.clear();
    if (!adj_list) return;
    auto it = adj_list->find(user);
    if (it == adj_list->end()) return;
    const vector<int>& friends = it->second;
    unordered_set<int> seen;
    for (int f : friends) {
        if (f == user) continue;
        if (seen.insert(f).second) out.push_back(f);
        if ((int)out.size() >= candidate_limit) return;
        auto it2 = adj_list->find(f);
        if (it2 == adj_list->end()) continue;
        for (int ff : it2->second) {
            if (ff == user) continue;
            if (seen.insert(ff).second) {
                out.push_back(ff);
                if ((int)out.size() >= candidate_limit) return;
            }
        }
    }
}

vector<pair<int,float>> Recommender::recommend_graph_registration(int user, int topk, int candidate_limit) const
{
    vector<pair<int,float>> out;
    if ((!profiles && !user_feats) || !adj_list) return out;

    if (profiles) {
        auto itq = profiles->find(user);
        if (itq == profiles->end()) return out;
        const UserProfile &q = itq->second;

        vector<int> candidates;
        gather_candidates_local(adj_list, user, candidates, candidate_limit);

        unordered_set<int> existing;
        auto itne = adj_list->find(user);
        if (itne != adj_list->end()) for (int v : itne->second) existing.insert(v);
        existing.insert(user);

        for (int c : candidates) {
            if (existing.find(c) != existing.end()) continue;
            auto itc = profiles->find(c);
            if (itc == profiles->end()) continue;
            float s = profile_similarity(q, itc->second);
            out.emplace_back(c, s);
        }
    } else {
        auto itq = user_feats->find(user);
        if (itq == user_feats->end()) return out;
        const auto &qvec = itq->second;
        vector<int> candidates;
        gather_candidates_local(adj_list, user, candidates, candidate_limit);
        unordered_set<int> existing;
        auto itne = adj_list->find(user);
        if (itne != adj_list->end()) for (int v : itne->second) existing.insert(v);
        existing.insert(user);
        for (int c : candidates) {
            if (existing.find(c) != existing.end()) continue;
            auto itc = user_feats->find(c);
            if (itc == user_feats->end()) continue;
            const auto &cvec = itc->second;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (auto &pa : qvec) na += (double)pa.second * pa.second;
            for (auto &pb : cvec) nb += (double)pb.second * pb.second;
            if (na > 0 && nb > 0) {
                if (qvec.size() < cvec.size()) {
                    for (auto &pa : qvec) {
                        auto jt = cvec.find(pa.first);
                        if (jt != cvec.end()) dot += (double)pa.second * jt->second;
                    }
                } else {
                    for (auto &pb : cvec) {
                        auto it2 = qvec.find(pb.first);
                        if (it2 != qvec.end()) dot += (double)pb.second * it2->second;
                    }
                }
                double denom = sqrt(na) * sqrt(nb);
                if (denom > 0.0) out.emplace_back(c, (float)(dot/denom));
                else out.emplace_back(c, 0.0f);
            } else {
                out.emplace_back(c, 0.0f);
            }
        }
    }

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_collaborative(int user, int topk, int candidate_limit) const
{
    vector<pair<int,float>> out;
    if ((!profiles && !user_feats) || !adj_list) return out;

    vector<int> friends;
    auto itf = adj_list->find(user);
    if (itf != adj_list->end()) friends = itf->second;

    vector<int> candidates;
    unordered_set<int> candidate_set;
    for (int f : friends) {
        auto itff = adj_list->find(f);
        if (itff == adj_list->end()) continue;
        for (int fof : itff->second) {
            if (fof == user) continue;
            if (candidate_set.insert(fof).second) candidates.push_back(fof);
            if ((int)candidates.size() >= candidate_limit) break;
        }
        if ((int)candidates.size() >= candidate_limit) break;
    }

    unordered_map<int,float> sim_u_f;
    if (profiles) {
        auto itq = profiles->find(user);
        if (itq == profiles->end()) return out;
        const UserProfile &q = itq->second;
        for (int f : friends) {
            auto itpf = profiles->find(f);
            if (itpf == profiles->end()) continue;
            sim_u_f[f] = profile_similarity(q, itpf->second);
        }
    } else {
        auto itq = user_feats->find(user);
        if (itq == user_feats->end()) return out;
        const auto &qvec = itq->second;
        for (int f : friends) {
            auto itpf = user_feats->find(f);
            if (itpf == user_feats->end()) continue;
            const auto &fvec = itpf->second;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (auto &pa : qvec) na += (double)pa.second * pa.second;
            for (auto &pb : fvec) nb += (double)pb.second * pb.second;
            if (na > 0 && nb > 0) {
                if (qvec.size() < fvec.size()) {
                    for (auto &pa : qvec) {
                        auto jt = fvec.find(pa.first);
                        if (jt != fvec.end()) dot += (double)pa.second * jt->second;
                    }
                } else {
                    for (auto &pb : fvec) {
                        auto it2 = qvec.find(pb.first);
                        if (it2 != qvec.end()) dot += (double)pb.second * it2->second;
                    }
                }
                double denom = sqrt(na) * sqrt(nb);
                if (denom > 0.0) sim_u_f[f] = (float)(dot/denom);
                else sim_u_f[f] = 0.0f;
            } else sim_u_f[f] = 0.0f;
        }
    }

    for (int cand : candidates) {
        if (cand == user) continue;
        double score = 0.0;
        if (profiles) {
            auto itpc = profiles->find(cand);
            if (itpc == profiles->end()) continue;
            for (int f : friends) {
                auto itsim = sim_u_f.find(f);
                if (itsim == sim_u_f.end()) continue;
                auto itpf = profiles->find(f);
                if (itpf == profiles->end()) continue;
                double s_f_fof = profile_similarity(itpf->second, itpc->second);
                score += (double)itsim->second * s_f_fof;
            }
        } else {
            auto itpc = user_feats->find(cand);
            if (itpc == user_feats->end()) continue;
            const auto &cvec = itpc->second;
            for (int f : friends) {
                auto itsim = sim_u_f.find(f);
                if (itsim == sim_u_f.end()) continue;
                auto itpf = user_feats->find(f);
                if (itpf == user_feats->end()) continue;
                const auto &fvec = itpf->second;
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (auto &pa : fvec) na += (double)pa.second * pa.second;
                for (auto &pb : cvec) nb += (double)pb.second * pb.second;
                double s_f_fof = 0.0;
                if (na > 0 && nb > 0) {
                    if (fvec.size() < cvec.size()) {
                        for (auto &pa : fvec) {
                            auto jt = cvec.find(pa.first);
                            if (jt != cvec.end()) dot += (double)pa.second * jt->second;
                        }
                    } else {
                        for (auto &pb : cvec) {
                            auto it2 = fvec.find(pb.first);
                            if (it2 != fvec.end()) dot += (double)pb.second * it2->second;
                        }
                    }
                    double denom = sqrt(na) * sqrt(nb);
                    if (denom > 0.0) s_f_fof = dot/denom;
                }
                score += (double)itsim->second * s_f_fof;
            }
        }
        out.emplace_back(cand, (float)score);
    }

    sort(out.begin(), out.end(), [](const pair<int,float>& A, const pair<int,float>& B){
        if (A.second == B.second) return A.first < B.first;
        return A.second > B.second;
    });
    if ((int)out.size() > topk) out.resize(topk);
    return out;
}

vector<pair<int,float>> Recommender::recommend_by_interest(int user, int topk, int candidate_limit) const
{
    return recommend_graph_registration(user, topk, candidate_limit);
}

vector<pair<int,float>> Recommender::recommend_friends_graph(int user, int topk, int candidate_limit) const {
    return recommend_graph_registration(user, topk, candidate_limit);
}
vector<pair<int,float>> Recommender::recommend_friends_collab(int user, int topk, int candidate_limit) const {
    return recommend_collaborative(user, topk, candidate_limit);
}
vector<pair<int,float>> Recommender::recommend_friends_by_interest(int user, int topk, int candidate_limit) const {
    return recommend_by_interest(user, topk, candidate_limit);
}
