#include "recommender.h"
#include "user_profile.h"

#include <cmath>
#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>

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

float Recommender::profile_similarity(const UserProfile &A, const UserProfile &B, const vector<string> &text_columns) const
{
    const int NUM_FIXED = 7; // public, gender, completion, age, region, clubs, friends
    int total_possible = NUM_FIXED + (int)text_columns.size();

    int used = 0;              
    double sum_Si = 0.0;        

    auto sigmoid = [](double x)->double {
        if (x >= 0) {
            double e = exp(-x);
            return 1.0 / (1.0 + e);
        } else {
            double e = exp(x);
            return e / (1.0 + e);
        }
    };

    auto compute_z = [&](const string &field_key, double s)->double {
        auto it = field_normalizers.find(field_key);
        if (it != field_normalizers.end() && it->second.second > 0.0) {
            double mean = it->second.first;
            double sd = it->second.second;
            return (s - mean) / sd;
        }
        return 6.0 * (s - 0.5);
    };

    // PUBLIC
    if (A.public_flag >= 0 && B.public_flag >= 0) {
        double s_pub = (A.public_flag == B.public_flag) ? 1.0 : 0.0;
        double z = compute_z("public", s_pub);
        sum_Si += sigmoid(z);
        ++used;
    }

    // GENDER
    if (A.gender >= 0 && B.gender >= 0) {
        double s_gen = (A.gender == B.gender) ? 1.0 : 0.0;
        double z = compute_z("gender", s_gen);
        sum_Si += sigmoid(z);
        ++used;
    }

    // COMPLETION (ratio)
    if (A.completion_percentage > 0 && B.completion_percentage > 0) {
        int amin = min(A.completion_percentage, B.completion_percentage);
        int amax = max(A.completion_percentage, B.completion_percentage);
        double s_comp = (amax > 0) ? ((double)amin / (double)amax) : 0.0;
        double z = compute_z("completion", s_comp);
        sum_Si += sigmoid(z);
        ++used;
    }

    // AGE (ratio)
    if (A.age > 0 && B.age > 0) {
        int amin = min(A.age, B.age);
        int amax = max(A.age, B.age);
        double s_age = (amax > 0) ? ((double)amin / (double)amax) : 0.0;
        double z = compute_z("age", s_age);
        sum_Si += sigmoid(z);
        ++used;
    }

    // REGION
    {
        // consider region non-empty if at least one part >=0 for both
        bool nonemptyA = (A.region_parts[0] >= 0 || A.region_parts[1] >= 0 || A.region_parts[2] >= 0);
        bool nonemptyB = (B.region_parts[0] >= 0 || B.region_parts[1] >= 0 || B.region_parts[2] >= 0);
        if (nonemptyA && nonemptyB) {
            double s_reg = region_similarity_local(A.region_parts, B.region_parts);
            double z = compute_z("region", s_reg);
            sum_Si += sigmoid(z);
            ++used;
        }
    }

    // CLUBS
    if (!A.clubs.empty() && !B.clubs.empty()) {
        double s_clubs = vec_set_similarity(A.clubs, B.clubs);
        double z = compute_z("clubs", s_clubs);
        sum_Si += sigmoid(z);
        ++used;
    }

    // FRIENDS
    if (!A.friends.empty() && !B.friends.empty()) {
        double s_friends = vec_set_similarity(A.friends, B.friends);
        double z = compute_z("friends", s_friends);
        sum_Si += sigmoid(z);
        ++used;
    }

    // TEXT COLUMNS
    for (size_t t = 0; t < text_columns.size(); ++t) {
        bool nonemptyA = (t < A.token_cols.size() && !A.token_cols[t].empty());
        bool nonemptyB = (t < B.token_cols.size() && !B.token_cols[t].empty());
        if (!nonemptyA || !nonemptyB) continue;
        const string &colname = text_columns[t];
        double s_text = 0.0;
        auto itidf = idf_per_col.find(colname);
        if (itidf != idf_per_col.end()) {
            s_text = tfidf_cosine_for_column(A.token_cols[t], B.token_cols[t], itidf->second);
        } else {
            s_text = cosine_counts_maps_local(A.token_cols[t], B.token_cols[t]);
        }
        auto itcol = column_normalizers.find(colname);
        double z;
        if (itcol != column_normalizers.end() && itcol->second.second > 0.0) {
            z = (s_text - itcol->second.first) / itcol->second.second;
        } else {
            z = 6.0 * (s_text - 0.5);
        }
        sum_Si += sigmoid(z);
        ++used;
    }

    if (used == 0) return 0.0f;

    double S = sum_Si / (double)used;
    double F = (double)used / (double)total_possible;

    if (S <= 0.0 && F <= 0.0) return 0.0f;
    double fas = (2.0 * S * F) / (S + F);
    return (float)fas;
}

float Recommender::profile_similarity(const UserProfile &A, const UserProfile &B) const {
    return profile_similarity(A, B, text_columns_internal);
}

static void gather_candidates(const unordered_map<int, vector<int>>* adj_list, int user, vector<int>& out, int candidate_limit) {
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
        gather_candidates(adj_list, user, candidates, candidate_limit);

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
        gather_candidates(adj_list, user, candidates, candidate_limit);
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
