#ifndef TFIDF_INDEX_H
#define TFIDF_INDEX_H

#include <unordered_map>
#include <vector>
#include <string>

struct UserProfile;

struct TFIDFIndex {
    void build(const std::unordered_map<int, UserProfile>& profiles, const std::vector<std::string>& text_columns);
    float weighted_cosine(const std::unordered_map<int,int>& A, const std::unordered_map<int,int>& B, int col_idx) const;
    void compute_tfidf_vector(const UserProfile& p, std::unordered_map<int,float>& out) const;
private:
    int N = 0;
    std::vector< std::unordered_map<int,int> > doc_freqs;
};

#endif
