#include "column_stats.h"
#include "user_profile.h"
#include <random>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

static float cosine_counts_maps_local(const unordered_map<int,int>& A, const unordered_map<int,int>& B) {
    if (A.empty() || B.empty()) return 0.0f;
    double dot = 0.0;
    double suma2 = 0.0;
    double sumb2 = 0.0;
    for (auto &pa : A) suma2 += (double)pa.second * (double)pa.second;
    for (auto &pb : B) sumb2 += (double)pb.second * (double)pb.second;
    if (suma2 <= 0.0 || sumb2 <= 0.0) return 0.0f;
    if (A.size() < B.size()) {
        for (auto it = A.begin(); it != A.end(); ++it) {
            int k = it->first;
            int va = it->second;
            auto jt = B.find(k);
            if (jt != B.end()) dot += (double)va * (double)(jt->second);
        }
    } else {
        for (auto it = B.begin(); it != B.end(); ++it) {
            int k = it->first;
            int vb = it->second;
            auto jt = A.find(k);
            if (jt != A.end()) dot += (double)vb * (double)(jt->second);
        }
    }
    double norm = sqrt(suma2) * sqrt(sumb2);
    if (norm <= 0.0) return 0.0f;
    return (float)(dot / norm);
}

vector<float> compute_column_mean_similarities(
    const unordered_map<int, UserProfile>& profiles_map,
    const vector<string>& text_columns,
    int sample_size,
    int comps_per_user
) {
    vector<float> sums(text_columns.size(), 0.0f);
    vector<int> counts(text_columns.size(), 0);
    vector<int> uids;
    uids.reserve(profiles_map.size());
    for (auto &p : profiles_map) uids.push_back(p.first);
    if (uids.empty()) return vector<float>(text_columns.size(), 0.0f);
    random_device rd;
    mt19937 gen(rd());
    shuffle(uids.begin(), uids.end(), gen);
    int take = (int)uids.size();
    if (sample_size > 0 && sample_size < take) take = sample_size;
    for (int i = 0; i < take; ++i) {
        int uid = uids[i];
        const UserProfile &A = profiles_map.at(uid);
        uniform_int_distribution<> dist(0, (int)uids.size() - 1);
        int made = 0;
        int tries = 0;
        while (made < comps_per_user && tries < comps_per_user * 10) {
            ++tries;
            int idx = dist(gen);
            if (uids[idx] == uid) continue;
            const UserProfile &B = profiles_map.at(uids[idx]);
            for (size_t t = 0; t < text_columns.size(); ++t) {
                const unordered_map<int,int> *pa = nullptr, *pb = nullptr;
                if (t < A.token_cols.size()) pa = &A.token_cols[t];
                if (t < B.token_cols.size()) pb = &B.token_cols[t];
                if (pa && pb && !pa->empty() && !pb->empty()) {
                    float s = cosine_counts_maps_local(*pa, *pb);
                    sums[t] += s;
                    counts[t] += 1;
                }
            }
            ++made;
        }
        if ((i+1) % 1000 == 0) {
            cout << "[column_stats] processed " << (i+1) << " / " << take << " users\n";
        }
    }
    vector<float> res(text_columns.size(), 0.0f);
    for (size_t t = 0; t < text_columns.size(); ++t) {
        if (counts[t] > 0) res[t] = sums[t] / (float)counts[t];
        else res[t] = 0.0f;
    }
    return res;
}

bool load_column_normalizers_csv(const string& path,
                                 const vector<string>& text_columns,
                                 vector<float>& out_norms)
{
    out_norms.assign(text_columns.size(), 0.0f);
    ifstream in(path);
    if (!in.is_open()) return false;
    string header;
    if (!getline(in, header)) { in.close(); return false; }
    string line;
    unordered_map<string,float> mapvals;
    while (getline(in, line)) {
        if (line.empty()) continue;
        string key;
        string val;
        size_t pos = line.find(',');
        if (pos == string::npos) continue;
        key = line.substr(0,pos);
        val = line.substr(pos+1);
        // trim key
        size_t a = 0; while (a < key.size() && isspace((unsigned char)key[a])) ++a;
        size_t b = key.size(); while (b > a && isspace((unsigned char)key[b-1])) --b;
        key = key.substr(a, b - a);
        // trim val
        a = 0; while (a < val.size() && isspace((unsigned char)val[a])) ++a;
        b = val.size(); while (b > a && isspace((unsigned char)val[b-1])) --b;
        val = val.substr(a, b - a);
        float f = 0.0f;
        try { f = (float) stof(val); } catch (...) { f = 0.0f; }
        mapvals[key] = f;
    }
    in.close();
    for (size_t i = 0; i < text_columns.size(); ++i) {
        const string &k = text_columns[i];
        if (mapvals.find(k) != mapvals.end()) out_norms[i] = mapvals[k];
        else out_norms[i] = 0.0f;
    }
    return true;
}
