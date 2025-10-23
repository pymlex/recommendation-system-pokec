#include "feature_extractor.h"
#include <cmath>
#include <sstream>
#include <algorithm>


using namespace std;


void FeatureExtractor::build_from_df(const vector<vector<string>>& df) {
    unordered_map<int, int> df_count;
    int doc_id = 0;

    for (size_t r = 0; r < df.size(); ++r) {
        const vector<string>& row = df[r];
        if (row.empty()) 
            continue;

        int uid = atoi(row[0].c_str());
        unordered_map<int, int> local;
        for (size_t c = 2; c < row.size(); ++c) {
            string cell = row[c];
            stringstream ss(cell);
            string token;

            while (ss >> token) {
                if (vocab.find(token) == vocab.end()) {
                    int id = (int) vocab.size();
                    vocab[token] = id;
                }

                int tid = vocab[token];
                local[tid] += 1;
            }
        }

        for (auto it = local.begin(); it != local.end(); ++it) 
            df_count[it->first] += 1;
        unordered_map<int,float> tf;
        for (auto it = local.begin(); it != local.end(); ++it) 
            tf[it->first] = (float) it->second;
        user_tfidf[uid] = tf;
        ++doc_id;
    }

    int N = (int) df.size();
    for (auto uit = user_tfidf.begin(); uit != user_tfidf.end(); ++uit) {
        unordered_map<int,float> v;
        float sum2 = 0.0f;
        for (auto it = uit->second.begin(); it != uit->second.end(); ++it) {
            int term = it->first;
            float tf = it->second;
            int dfc = 1;
            if (df_count.find(term) != df_count.end()) dfc = df_count[term];
            float idf = log( (float) N / (1.0f + (float) dfc) );
            float val = tf * idf;
            v[term] = val;
            sum2 += val * val;
        }
        float norm = sqrt(sum2);
        if (norm > 0.0f)
        {
            for (auto it = v.begin(); it != v.end(); ++it) it->second /= norm;
        }
        user_tfidf[uit->first] = v;
    }
}

float FeatureExtractor::cosine_between(const unordered_map<int,float>& a, const unordered_map<int,float>& b) {
    float dot = 0.0f;
    if (a.size() < b.size()) {
        for (auto it = a.begin(); it != a.end(); ++it) {
            int k = it->first;
            float va = it->second;
            auto jt = b.find(k);
            if (jt != b.end()) dot += va * jt->second;
        }
    } else {
        for (auto it = b.begin(); it != b.end(); ++it) {
            int k = it->first;
            float vb = it->second;
            auto jt = a.find(k);
            if (jt != a.end()) dot += vb * a.at(k);
        }
    }
    return dot;
}

vector<pair<int,float>> FeatureExtractor::cosine_sim_sparse(const unordered_map<int,float>& a, int topk) {
    vector<pair<int,float>> scores;
    for (auto it = user_tfidf.begin(); it != user_tfidf.end(); ++it) {
        int uid = it->first;
        float s = cosine_between(a, it->second);
        scores.push_back(make_pair(uid, s));
    }

    sort(scores.begin(), scores.end(), 
     [](const pair<int, float>& A, const pair<int, float>& B) {
         if (A.second == B.second) {
             return A.first < B.first; 
         }
         return A.second > B.second; 
     });
    
    if ((int) scores.size() > topk) 
        scores.resize(topk);
    return scores;
}
