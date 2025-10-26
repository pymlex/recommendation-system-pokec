#include "utils.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

unordered_map<int, vector<int>> build_adj_list(const unordered_map<int, vector<pair<int,float>>>& adj_weighted) {
    unordered_map<int, vector<int>> out;
    for (auto it = adj_weighted.begin(); it != adj_weighted.end(); ++it) {
        int u = it->first;
        const vector<pair<int,float>>& vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i) out[u].push_back(vec[i].first);
    }
    return out;
}

static float compute_dot(const unordered_map<int,float>& a, const unordered_map<int,float>& b) {
    float dot = 0.0f;
    if (a.size() < b.size()) {
        for (auto &pa : a) {
            auto it = b.find(pa.first);
            if (it != b.end()) dot += pa.second * it->second;
        }
    } else {
        for (auto &pb : b) {
            auto it = a.find(pb.first);
            if (it != a.end()) dot += pb.second * it->second;
        }
    }
    return dot;
}

float evaluate_holdout_hit_at_k(const unordered_map<int, vector<int>>& adj_list,
                                const unordered_map<int, unordered_map<int,float>>& feats,
                                int sample_size, int k) {
    vector<int> users;
    for (auto it = adj_list.begin(); it != adj_list.end(); ++it) users.push_back(it->first);
    if (users.empty()) return 0.0f;
    random_device rd;
    mt19937 gen(rd());
    shuffle(users.begin(), users.end(), gen);
    int take = sample_size;
    if (take > (int)users.size()) take = (int)users.size();
    int hits = 0;
    for (int i = 0; i < take; ++i) {
        int u = users[i];
        const vector<int>& neigh = adj_list.at(u);
        if (neigh.empty()) continue;
        int held = neigh[0];
        vector<int> reduced;
        for (size_t j = 1; j < neigh.size(); ++j) reduced.push_back(neigh[j]);
        const auto qit = feats.find(u);
        if (qit == feats.end()) continue;
        const unordered_map<int,float>& q = qit->second;
        vector<pair<int,float>> scores;
        for (auto it = feats.begin(); it != feats.end(); ++it) {
            int cand = it->first;
            if (cand == u) continue;
            bool skip = false;
            for (size_t z = 0; z < reduced.size(); ++z) if (reduced[z] == cand) { skip = true; break; }
            if (skip) continue;
            float dot = compute_dot(q, it->second);
            scores.push_back(make_pair(cand, dot));
        }
        sort(scores.begin(), scores.end(), [](const pair<int,float>& A, const pair<int,float>& B){
            if (A.second == B.second) return A.first < B.first;
            return A.second > B.second;
        });
        int limit = k;
        if ((int)scores.size() < limit) limit = (int)scores.size();
        bool hit = false;
        for (int p = 0; p < limit; ++p) if (scores[p].first == held) { hit = true; break; }
        if (hit) ++hits;
    }
    return (float)hits / (float)take;
}

vector<string> load_text_columns_from_file(const string& path) {
    vector<string> out;
    ifstream in(path);
    if (!in.is_open()) return out;
    string line;
    while (getline(in, line)) {
        size_t a = 0;
        while (a < line.size() && isspace((unsigned char)line[a])) ++a;
        size_t b = line.size();
        while (b > a && isspace((unsigned char)line[b-1])) --b;
        if (b <= a) continue;
        string token = line.substr(a, b - a);
        if (token.empty()) continue;
        if (token[0] == '#') continue;
        out.push_back(token);
    }
    return out;
}
