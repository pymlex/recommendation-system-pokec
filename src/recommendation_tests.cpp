#include "recommendation_tests.h"
#include "user_profile.h"
#include "recommender.h"

#include <random>
#include <iostream>
#include <set>
#include <algorithm>
#include <unordered_set>

using namespace std;

void print_example_recommendations(const unordered_map<int, UserProfile>& profiles,
                                   const unordered_map<int, vector<int>>& adj_list,
                                   Recommender& rec,
                                   const unordered_map<int, string>& club_id_to_name,
                                   const vector<string>& text_columns)
{
    if (profiles.empty()) {
        cout << "[example] profiles empty\n";
        return;
    }
    int uid = -1;
    for (auto &kv : profiles) {
        if (adj_list.find(kv.first) != adj_list.end()) { uid = kv.first; break; }
    }
    if (uid == -1) uid = profiles.begin()->first;

    uid = 35967; //set id as an option in this function

    cout << "=== Example recommendations for user " << uid << " ===\n";
    const UserProfile &p = profiles.at(uid);

    cout << "Existing friends (" << p.friends.size() << "): ";
    for (size_t i = 0; i < p.friends.size() && i < 50; ++i) cout << p.friends[i] << (i+1<p.friends.size() ? "," : "");
    cout << "\n";

    cout << "Non-empty properties:\n";
    if (p.age > 0) cout << "  age=" << p.age << "\n";
    if (p.gender >= 0) cout << "  gender=" << p.gender << "\n";
    if (!p.clubs.empty()) {
        cout << "  clubs (" << p.clubs.size() << "):\n";
        for (auto cid : p.clubs) {
            auto it = club_id_to_name.find((int)cid);
            cout << "    " << cid << " : " << (it != club_id_to_name.end() ? it->second : string("<name?>")) << "\n";
        }
    }

    cout << "\nTop friend recs (graph-registration):\n";
    auto friends_rec = rec.recommend_graph_registration(uid, 10);
    for (auto &pr : friends_rec) cout << "  user " << pr.first << " score=" << pr.second << "\n";

    cout << "\nTop club recs (collaborative):\n";
    auto club_rec = rec.recommend_clubs_collab(uid, 20);
    for (auto &pr : club_rec) {
        int cid = pr.first;
        auto it = club_id_to_name.find(cid);
        cout << "  club " << cid << " score=" << pr.second << " name=" << (it!=club_id_to_name.end() ? it->second : string("<name?>")) << "\n";
    }

    cout << "\nTop friend recs (by-interest):\n";
    auto interest_rec = rec.recommend_by_interest(uid, 10);
    for (auto &pr : interest_rec) cout << "  user " << pr.first << " score=" << pr.second << "\n";

    cout << "=== End example ===\n\n";
}

RecommendTestMetrics run_recommendation_tests_sample(const unordered_map<int, UserProfile>& profiles,
                                                     const unordered_map<int, vector<int>>& adj_list,
                                                     const unordered_map<int, string>& club_id_to_name,
                                                     Recommender& base_rec,
                                                     const vector<string>& text_columns,
                                                     int sample_size,
                                                     int topk)
{
    RecommendTestMetrics metrics;
    if (profiles.empty() || adj_list.empty()) return metrics;

    vector<int> all;
    for (auto &kv : profiles) all.push_back(kv.first);
    mt19937 rng(1234567);
    shuffle(all.begin(), all.end(), rng);

    int taken = 0;
    int hits_graph = 0;
    int hits_collab = 0;
    int hits_interest = 0;
    double total_club_prec = 0.0;
    double total_club_rec = 0.0;
    int club_users_counted = 0;

    int c = 0;
    for (int uid : all) {
        if (!(c%10)) {
            cout << "Processed " << c << " samples" << endl;
        }
        c++;

        if (taken >= sample_size) break;
        auto itadj = adj_list.find(uid);
        if (itadj == adj_list.end()) continue;
        const auto &friends = itadj->second;
        if (friends.size() < 4) continue; 
        int hold_k = max(1, (int)friends.size() / 4);
        vector<int> idx(friends.size());
        for (size_t i = 0; i < friends.size(); ++i) idx[i] = (int)i;
        shuffle(idx.begin(), idx.end(), rng);
        unordered_set<int> held;
        for (int i = 0; i < hold_k; ++i) held.insert(friends[idx[i]]);

        unordered_map<int, vector<int>> adj_mod = adj_list;
        vector<int> newf;
        for (int f : friends) if (held.find(f) == held.end()) newf.push_back(f);
        adj_mod[uid] = newf;

        Recommender rec(&profiles, &adj_mod);
        rec.set_field_normalizers(base_rec.field_normalizers);
        rec.set_column_normalizers(base_rec.column_normalizers);
        rec.set_text_columns(text_columns);
        rec.set_tfidf_index(base_rec.idf_per_col);

        auto out_g = rec.recommend_graph_registration(uid, topk, 5000);
        bool hitg = false;
        int found_g = 0;
        for (auto &p : out_g) if (held.find(p.first) != held.end()) { hitg = true; ++found_g; }
        if (hitg) ++hits_graph;

        auto out_c = rec.recommend_collaborative(uid, topk, 5000);
        bool hitc = false;
        int found_c = 0;
        for (auto &p : out_c) if (held.find(p.first) != held.end()) { hitc = true; ++found_c; }
        if (hitc) ++hits_collab;

        auto out_i = rec.recommend_by_interest(uid, topk, 5000);
        bool hiti = false;
        int found_i = 0;
        for (auto &p : out_i) if (held.find(p.first) != held.end()) { hiti = true; ++found_i; }
        if (hiti) ++hits_interest;

        auto club_pred = rec.recommend_clubs_collab(uid, topk, 5000);
        const UserProfile &up = profiles.at(uid);
        unordered_set<int> actual_clubs;
        for (auto c : up.clubs) actual_clubs.insert((int)c);
        if (!actual_clubs.empty()) {
            int hit_club_count = 0;
            for (size_t i = 0; i < club_pred.size() && i < (size_t)topk; ++i) {
                if (actual_clubs.find(club_pred[i].first) != actual_clubs.end()) ++hit_club_count;
            }
            double prec = (double)hit_club_count / (double)topk;
            double rec = (double)hit_club_count / (double)actual_clubs.size();
            total_club_prec += prec;
            total_club_rec += rec;
            ++club_users_counted;
        }

        ++taken;
    }

    if (taken > 0) {
        metrics.graph_hit_rate = (double)hits_graph / (double)taken;
        metrics.collab_hit_rate = (double)hits_collab / (double)taken;
        metrics.interest_hit_rate = (double)hits_interest / (double)taken;
    }
    if (club_users_counted > 0) {
        metrics.avg_club_prec_at_k = total_club_prec / (double)club_users_counted;
        metrics.avg_club_recall_at_k = total_club_rec / (double)club_users_counted;
    }
    return metrics;
}
