#include "tfidf_index.h"
#include "user_profile.h"
#include <cmath>

using namespace std;

void TFIDFIndex::build(const unordered_map<int, UserProfile>& profiles, const vector<string>& text_columns) {
    N = (int)profiles.size();
    doc_freqs.clear();
    doc_freqs.resize(text_columns.size());
    for (auto &kv : profiles) {
        const UserProfile &p = kv.second;
        for (size_t t = 0; t < text_columns.size(); ++t) {
            if (t >= p.token_cols.size()) continue;
            for (auto &pr : p.token_cols[t]) {
                doc_freqs[t][pr.first] += 1;
            }
        }
    }
    // build idf_per_col using same ordering of text_columns
    idf_per_col.clear();
    for (size_t t = 0; t < text_columns.size(); ++t) {
        const auto &dfmap = doc_freqs[t];
        unordered_map<int,float> m;
        for (auto &pr : dfmap) {
            // idf: use smooth variant
            float idf = (float)log(1.0 + (double)N / (1.0 + (double)pr.second));
            m[pr.first] = idf;
        }
        idf_per_col[text_columns[t]] = std::move(m);
    }
}

static double idf_val_local(const unordered_map<int,int>& dfmap, int N, int token) {
    auto it = dfmap.find(token);
    int df = (it == dfmap.end()) ? 0 : it->second;
    return log(1.0 + (double)N / (1.0 + (double)df));
}

float TFIDFIndex::weighted_cosine(const unordered_map<int,int>& A, const unordered_map<int,int>& B, int col_idx) const {
    if (A.empty() || B.empty()) return 0.0f;
    if (col_idx < 0 || col_idx >= (int)doc_freqs.size()) return 0.0f;
    const auto &dfmap = doc_freqs[col_idx];
    double dot = 0.0;
    double suma2 = 0.0;
    double sumb2 = 0.0;
    for (auto &pa : A) {
        double idf = idf_val_local(dfmap, N, pa.first);
        double w = (double)pa.second * idf;
        suma2 += w*w;
    }
    for (auto &pb : B) {
        double idf = idf_val_local(dfmap, N, pb.first);
        double w = (double)pb.second * idf;
        sumb2 += w*w;
    }
    if (suma2 <= 0.0 || sumb2 <= 0.0) return 0.0f;
    if (A.size() < B.size()) {
        for (auto &pa : A) {
            auto itb = B.find(pa.first);
            if (itb == B.end()) continue;
            double w1 = (double)pa.second * idf_val_local(dfmap, N, pa.first);
            double w2 = (double)itb->second * idf_val_local(dfmap, N, itb->first);
            dot += w1 * w2;
        }
    } else {
        for (auto &pb : B) {
            auto ita = A.find(pb.first);
            if (ita == A.end()) continue;
            double w1 = (double)ita->second * idf_val_local(dfmap, N, ita->first);
            double w2 = (double)pb.second * idf_val_local(dfmap, N, pb.first);
            dot += w1 * w2;
        }
    }
    double norm = sqrt(suma2) * sqrt(sumb2);
    if (norm <= 0.0) return 0.0f;
    return (float)(dot / norm);
}

void TFIDFIndex::compute_tfidf_vector(const UserProfile& p, unordered_map<int,float>& out) const {
    out.clear();
    if (N <= 0) return;
    size_t T = doc_freqs.size();
    for (size_t t = 0; t < T && t < p.token_cols.size(); ++t) {
        const auto &dfmap = doc_freqs[t];
        for (auto &pr : p.token_cols[t]) {
            int token = pr.first;
            int tf = pr.second;
            double idf = idf_val_local(dfmap, N, token);
            double w = (double)tf * idf;
            out[token] += (float)w;
        }
    }
}
