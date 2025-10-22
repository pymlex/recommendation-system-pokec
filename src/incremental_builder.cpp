#include "incremental_builder.h"
#include "bin_reader.h"
#include <cmath>
using namespace std;

bool build_user_feats_from_bin(const string& bin_path, const unordered_map<int, pair<uint32_t,uint32_t> >& idx_map, const vector<int>& user_ids, unordered_map<int, unordered_map<int,float> >& out_feats)
{
    out_feats.clear();
    for (size_t i = 0; i < user_ids.size(); ++i)
    {
        int uid = user_ids[i];
        UserRecord rec;
        if (! read_user_record(bin_path, idx_map, uid, rec)) continue;
        unordered_map<int,float> feats;
        for (size_t ci = 0; ci < rec.token_cols.size(); ++ci)
        {
            const vector<pair<uint32_t,uint32_t>>& col = rec.token_cols[ci];
            for (size_t j = 0; j < col.size(); ++j)
            {
                uint32_t tid = col[j].first;
                uint32_t cnt = col[j].second;
                float prev = 0.0f;
                if (feats.find((int)tid) != feats.end()) prev = feats.at((int)tid);
                feats[(int)tid] = prev + (float) cnt;
            }
        }
        float sum2 = 0.0f;
        for (auto it = feats.begin(); it != feats.end(); ++it) sum2 += it->second * it->second;
        float norm = sqrt(sum2);
        if (norm > 0.0f)
        {
            for (auto it = feats.begin(); it != feats.end(); ++it) it->second = it->second / norm;
        }
        out_feats[uid] = feats;
    }
    return true;
}
