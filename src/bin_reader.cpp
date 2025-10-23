#include "bin_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>


using namespace std;


bool load_index_map(const string& idx_path, unordered_map<int, pair<uint32_t,uint32_t> >& out_idx) {
    out_idx.clear();
    ifstream in(idx_path);

    if (! in.is_open()) 
        return false;

    string line;
    while (getline(in, line)) {
        if (line.empty()) 
            continue;
        stringstream ss(line);
        string a,b,c;
        if (! (getline(ss, a, ',') && getline(ss, b, ',') && getline(ss, c))) 
            continue;
        int uid = atoi(a.c_str());
        uint32_t off = (uint32_t) atoi(b.c_str());
        uint32_t len = (uint32_t) atoi(c.c_str());
        out_idx[uid] = make_pair(off, len);
    }

    return true;
}

static string read_string_len(ifstream& in, uint32_t len) {
    if (len == 0) 
        return string();

    string s;
    s.resize(len);
    in.read(&s[0], (streamsize) len);

    return s;
}

bool read_user_record(const string& bin_path, 
                      const unordered_map<int, pair<uint32_t,uint32_t> >& idx_map, 
                      int user_id, 
                      UserRecord& out_rec) {
    out_rec = UserRecord();

    if (idx_map.find(user_id) == idx_map.end()) 
        return false;

    pair<uint32_t,uint32_t> pr = idx_map.at(user_id);
    uint32_t offset = pr.first;
    ifstream in(bin_path, ios::binary);
    if (! in.is_open()) 
        return false;

    in.seekg((streamoff) offset);
    uint32_t uid = 0;
    in.read(reinterpret_cast<char*>(&uid), sizeof(uint32_t));
    out_rec.user_id = uid;
    uint32_t glen = 0;
    in.read(reinterpret_cast<char*>(&glen), sizeof(uint32_t));
    out_rec.gender = read_string_len(in, glen);
    uint32_t rlen = 0;
    in.read(reinterpret_cast<char*>(&rlen), sizeof(uint32_t));
    out_rec.region = read_string_len(in, rlen);
    uint32_t alen = 0;
    in.read(reinterpret_cast<char*>(&alen), sizeof(uint32_t));
    out_rec.age = read_string_len(in, alen);
    uint32_t clubs_count = 0;
    in.read(reinterpret_cast<char*>(&clubs_count), sizeof(uint32_t));

    for (uint32_t i = 0; i < clubs_count; ++i) {
        uint32_t cid = 0;
        in.read(reinterpret_cast<char*>(&cid), sizeof(uint32_t));
        out_rec.clubs.push_back(cid);
    }

    uint32_t cols_num = 0;
    in.read(reinterpret_cast<char*>(&cols_num), sizeof(uint32_t));
    for (uint32_t ci = 0; ci < cols_num; ++ci) {
        uint32_t pairs_count = 0;
        in.read(reinterpret_cast<char*>(&pairs_count), sizeof(uint32_t));
        vector<pair<uint32_t,uint32_t>> vec;
        for (uint32_t pi = 0; pi < pairs_count; ++pi) {
            uint32_t tid = 0;
            uint32_t cnt = 0;
            in.read(reinterpret_cast<char*>(&tid), sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(&cnt), sizeof(uint32_t));
            vec.push_back(make_pair(tid, cnt));
        }
        out_rec.token_cols.push_back(vec);
    }
    
    return true;
}
