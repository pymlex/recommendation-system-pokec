#ifndef BIN_READER_H
#define BIN_READER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

using namespace std;

struct UserRecord {
    uint32_t user_id;
    uint32_t ispublic;
    uint32_t completion_percentage;
    uint32_t gender;
    vector<uint32_t> region; 
    uint32_t age;   
    vector<uint32_t> clubs;
    vector<vector<pair<uint32_t,uint32_t>>> token_cols;
};

bool load_index_map(const string& idx_path,
                    unordered_map<int, pair<uint32_t,uint32_t> >& out_idx);
bool read_user_record(const string& bin_path,
                      const unordered_map<int, pair<uint32_t,uint32_t> >& idx_map,
                      int user_id, UserRecord& out_rec);

#endif
