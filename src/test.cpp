#include "test.h"
#include "recommender.h"
#include "user_profile.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <iomanip>

using namespace std;

void run_friends_holdout_test(const unordered_map<int, UserProfile>& profiles,
                              const unordered_map<int, vector<int>>& adj_list,
                              const vector<string>& text_columns,
                              const Recommender& base_rec,
                              int sample_size,
                              const string& out_path)
{
    vector<int> candidates;
    for (auto &kv : profiles) {
        int uid = kv.first;
        auto it = adj_list.find(uid);
        if (it == adj_list.end()) continue;
        if ((int)it->second.size() >= 20) candidates.push_back(uid);
    }
    if (candidates.empty()) {
        cout << "[test] no suitable users found\n";
        return;
    }

    mt19937 rng(1234567);
    shuffle(candidates.begin(), candidates.end(), rng);

    unordered_map<int, vector<int>> adj_mod = adj_list;
    Recommender rec(&profiles, &adj_mod);
    rec.set_field_normalizers(base_rec.field_normalizers);
    rec.set_column_normalizers(base_rec.column_normalizers);
    rec.set_text_columns(text_columns);
    rec.set_tfidf_index(base_rec.idf_per_col);

    vector<double> results;
    results.reserve(sample_size);

    cout << "[test] Testing the system (friends recommender mode)\n";

    int taken = 0;
    int processed = 0;
    for (int uid : candidates) {
        if (taken >= sample_size) break;
        ++processed;

        auto itadj = adj_list.find(uid);
        if (itadj == adj_list.end()) continue;
        const vector<int> &friends = itadj->second;
        int F = (int)friends.size();
        if (F < 2) continue;

        int hold_k = F / 5;
        if (hold_k <= 0) continue;

        vector<int> idx(F);
        for (int i = 0; i < F; ++i) idx[i] = i;
        shuffle(idx.begin(), idx.end(), rng);

        unordered_set<int> held;
        for (int i = 0; i < hold_k; ++i) held.insert(friends[idx[i]]);

        vector<int> newf;
        newf.reserve(F - hold_k);
        for (int f : friends) if (held.find(f) == held.end()) newf.push_back(f);

        adj_mod[uid] = std::move(newf);

        auto preds = rec.recommend_collaborative(uid, hold_k, 1000);

        int hits = 0;
        for (size_t i = 0; i < preds.size() && (int)i < hold_k; ++i) {
            if (held.find(preds[i].first) != held.end()) ++hits;
        }

        double ratio = (hold_k > 0) ? (double)hits / (double)hold_k : 0.0;
        results.push_back(ratio);
        ++taken;

        if ((processed % 5) == 0) {
            cout << "[test] processed " << processed << " candidates, collected " << taken << " samples\n";
        }
    }

    ofstream out(out_path);
    if (! out.is_open()) {
        cout << "[test] cannot open output file: " << out_path << "\n";
        return;
    }
    out << fixed << setprecision(6);
    for (double v : results) out << v << "\n";
    out.close();

    double sum = 0.0;
    for (double v : results) sum += v;
    double avg = results.empty() ? 0.0 : sum / (double)results.size();
    cout << "[test] finished. users tested: " << results.size()
         << " average_ratio=" << avg << " saved to " << out_path << "\n";
}
