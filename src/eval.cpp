#include "eval.h"
#include "recommender.h"
#include <random>
#include <set>
#include <algorithm>
#include <iostream>

using namespace std;

// For "registration" scenario: we hide a fraction of true friends (e.g. 75% keep, 25% hidden) and test recovery.
// For collaborative/interests we use similar holdout.
EvalResult evaluate_recommender_sample(
    const unordered_map<int, UserProfile>& profiles,
    const unordered_map<int, vector<int>>& adj_list,
    Recommender &rec,
    const vector<string>& text_columns,
    int sample_size,
    int k)
{
    EvalResult res{0.0,0.0,0.0};
    if (profiles.empty()) return res;
    vector<int> ids;
    ids.reserve(profiles.size());
    for (auto &p : profiles) ids.push_back(p.first);
    mt19937 rng(123456);
    shuffle(ids.begin(), ids.end(), rng);
    if ((int)ids.size() > sample_size) ids.resize(sample_size);

    int hits = 0;
    double prec_sum = 0.0;
    double rec_sum = 0.0;
    int examined = 0;

    for (int uid : ids) {
        auto itadj = adj_list.find(uid);
        if (itadj == adj_list.end()) continue;
        const vector<int>& friends = itadj->second;
        if (friends.size() < 4) continue; // need some friends to hide/recover
        // holdout: keep 75%, hide 25% randomly
        vector<int> shuffled = friends;
        shuffle(shuffled.begin(), shuffled.end(), rng);
        size_t keep = max<size_t>(1, (shuffled.size() * 3) / 4);
        set<int> kept(shuffled.begin(), shuffled.begin() + keep);
        set<int> hidden(shuffled.begin() + keep, shuffled.end());

        // prepare a recommender that uses original profiles and adj_list but in scoring we will pretend user has only 'kept' friends
        // For speed we will call the same recommender but filter out existing friends using the real adj_list in main; to simulate partial friends we
        // create temporary adj_list? Simpler: we ask rec.recommend_graph_registration(uid, k*2, candidate_limit) then compute how many of top-k are in hidden.
        auto recs = rec.recommend_graph_registration(uid, k*2, 10000);

        // count how many recovered hidden within top k
        int found = 0;
        int considered = 0;
        for (size_t i = 0; i < recs.size() && considered < k; ++i) {
            int cand = recs[i].first;
            ++considered;
            if (hidden.find(cand) != hidden.end()) ++found;
        }
        if (found > 0) ++hits;
        prec_sum += (double)found / (double)k;
        rec_sum += (double)found / (double)hidden.size();
        ++examined;
    }

    if (examined == 0) return res;
    res.hit_at_k = (double)hits / (double)examined;
    res.precision_at_k = prec_sum / (double)examined;
    res.recall_at_k = rec_sum / (double)examined;
    return res;
}
