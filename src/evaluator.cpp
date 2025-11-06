#include "evaluator.h"
#include "recommender.h"
#include "tfidf_index.h"
#include <random>
#include <set>

using namespace std;

EvalMetrics evaluate_recommenders_holdout(const unordered_map<int, UserProfile>& profiles,
                                         const unordered_map<int, vector<int>>& adj_list,
                                         const vector<string>& text_columns,
                                         int sample_size,
                                         int topk,
                                         const unordered_map<int, unordered_map<int,float>>* super_feats)
{
    EvalMetrics res;
    if (profiles.empty()) return res;
    vector<int> all;
    for (auto &kv : profiles) all.push_back(kv.first);
    mt19937 rng(123456);
    shuffle(all.begin(), all.end(), rng);

    vector<int> test_users;
    for (int uid : all) {
        auto it = adj_list.find(uid);
        if (it == adj_list.end()) continue;
        if ((int)it->second.size() >= 4) test_users.push_back(uid);
        if ((int)test_users.size() >= sample_size) break;
    }
    if (test_users.empty()) return res;

    TFIDFIndex tfidf;
    tfidf.build(profiles, text_columns);

    int hits_g = 0, hits_c = 0, hits_i = 0, hits_s = 0, tot = 0;
    for (int uid : test_users) {
        const auto &friends = adj_list.at(uid);
        if (friends.size() < 4) continue;
        int hold_k = max(1, (int)friends.size() / 4);
        vector<int> idx(friends.size());
        for (size_t i = 0; i < friends.size(); ++i) idx[i] = (int)i;
        shuffle(idx.begin(), idx.end(), rng);
        set<int> held;
        for (int i = 0; i < hold_k; ++i) held.insert(friends[idx[i]]);

        unordered_map<int, vector<int>> adj_mod = adj_list;
        vector<int> newf;
        for (int f : adj_list.at(uid)) if (held.find(f) == held.end()) newf.push_back(f);
        adj_mod[uid] = newf;

        Recommender rec((unordered_map<int, UserProfile>*)&profiles, &adj_mod);
        rec.set_text_columns(text_columns);
        rec.set_tfidf_index(&tfidf);

        auto out_g = rec.recommend_friends_graph(uid, topk, 5000);
        bool hitg = false;
        for (auto &p : out_g) if (held.find(p.first) != held.end()) { hitg = true; break; }
        if (hitg) ++hits_g;

        auto out_c = rec.recommend_friends_collab(uid, topk, 5000);
        bool hitc = false;
        for (auto &p : out_c) if (held.find(p.first) != held.end()) { hitc = true; break; }
        if (hitc) ++hits_c;

        auto out_i = rec.recommend_friends_by_interest(uid, topk, 5000);
        bool hiti = false;
        for (auto &p : out_i) if (held.find(p.first) != held.end()) { hiti = true; break; }
        if (hiti) ++hits_i;

        if (super_feats) {
            // create a recommender using user_tfidf built on the fly
            // Build temporary user_tfidf map for the recommender (only references needed in recommend_from_supernodes)
            unordered_map<int, unordered_map<int,float>> temp_user_tfidf;
            TFIDFIndex tmp_tfidf;
            tmp_tfidf.build(profiles, text_columns);
            for (auto &kv : profiles) {
                unordered_map<int,float> vec;
                tmp_tfidf.compute_tfidf_vector(kv.second, vec);
                if (!vec.empty()) temp_user_tfidf[kv.first] = std::move(vec);
            }
            Recommender rec_t(&temp_user_tfidf, &adj_mod);
            rec_t.set_text_columns(text_columns);
            rec_t.set_tfidf_index(&tmp_tfidf);
            auto out_s = rec_t.recommend_from_supernodes(uid, *super_feats, topk);
            bool hits = false;
            for (auto &p : out_s) if (held.find(p.first) != held.end()) { hits = true; break; }
            if (hits) ++hits_s;
        }

        ++tot;
    }
    if (tot > 0) {
        res.graph_hit = (double)hits_g / tot;
        res.collab_hit = (double)hits_c / tot;
        res.interest_hit = (double)hits_i / tot;
        if (super_feats) res.supernode_hit = (double)hits_s / tot;
    }
    return res;
}
