#include "recommender.h"
#include "user_profile.h"

#include <cmath>
#include <unordered_map>
#include <string>

using namespace std;

float Recommender::profile_similarity(const UserProfile &A, const UserProfile &B, const vector<string> &text_columns) const
{
    const int NUM_FIXED = 7;
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

    if (A.public_flag >= 0 && B.public_flag >= 0) {
        double s_pub = (A.public_flag == B.public_flag) ? 1.0 : 0.0;
        double z = compute_z("public", s_pub);
        sum_Si += sigmoid(z);
        ++used;
    }

    if (A.gender >= 0 && B.gender >= 0) {
        double s_gen = (A.gender == B.gender) ? 1.0 : 0.0;
        double z = compute_z("gender", s_gen);
        sum_Si += sigmoid(z);
        ++used;
    }

    if (A.completion_percentage > 0 && B.completion_percentage > 0) {
        int amin = min(A.completion_percentage, B.completion_percentage);
        int amax = max(A.completion_percentage, B.completion_percentage);
        double s_comp = (amax > 0) ? ((double)amin / (double)amax) : 0.0;
        double z = compute_z("completion", s_comp);
        sum_Si += sigmoid(z);
        ++used;
    }

    if (A.age > 0 && B.age > 0) {
        int amin = min(A.age, B.age);
        int amax = max(A.age, B.age);
        double s_age = (amax > 0) ? ((double)amin / (double)amax) : 0.0;
        double z = compute_z("age", s_age);
        sum_Si += sigmoid(z);
        ++used;
    }

    bool nonemptyA = (A.region_parts[0] >= 0 || A.region_parts[1] >= 0 || A.region_parts[2] >= 0);
    bool nonemptyB = (B.region_parts[0] >= 0 || B.region_parts[1] >= 0 || B.region_parts[2] >= 0);
    if (nonemptyA && nonemptyB) {
        double s_reg = region_similarity_local(A.region_parts, B.region_parts);
        double z = compute_z("region", s_reg);
        sum_Si += sigmoid(z);
        ++used;
    }

    if (!A.clubs.empty() && !B.clubs.empty()) {
        double s_clubs = vec_set_similarity(A.clubs, B.clubs);
        double z = compute_z("clubs", s_clubs);
        sum_Si += sigmoid(z);
        ++used;
    }

    if (!A.friends.empty() && !B.friends.empty()) {
        double s_friends = vec_set_similarity(A.friends, B.friends);
        double z = compute_z("friends", s_friends);
        sum_Si += sigmoid(z);
        ++used;
    }

    for (size_t t = 0; t < text_columns.size(); ++t) {
        bool ta = (t < A.token_cols.size() && !A.token_cols[t].empty());
        bool tb = (t < B.token_cols.size() && !B.token_cols[t].empty());
        if (!ta || !tb) continue;
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
