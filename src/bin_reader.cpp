#include "bin_reader.h"
#include <fstream>
#include <sstream>

using namespace std;

bool load_index_map(const string& idx_path, unordered_map<int, pair<uint32_t,uint32_t> >& out_idx) {
    out_idx.clear();
    ifstream in(idx_path);

    if (!in.is_open())
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

    in.close();
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

    auto it = idx_map.find(user_id);
    if (it == idx_map.end())
        return false;

    pair<uint32_t,uint32_t> pr = it->second;
    uint32_t offset = pr.first;
    ifstream in(bin_path, ios::binary);
    if (! in.is_open())
        return false;

    in.seekg((streamoff) offset);

    // user_id
    uint32_t uid = 0;
    in.read(reinterpret_cast<char*>(&uid), sizeof(uint32_t));
    out_rec.user_id = uid;

    // ispublic
    uint32_t ispub = 0;
    in.read(reinterpret_cast<char*>(&ispub), sizeof(uint32_t));
    out_rec.ispublic = ispub;

    // completion_percentage
    uint32_t completion = 0;
    in.read(reinterpret_cast<char*>(&completion), sizeof(uint32_t));
    out_rec.completion_percentage = completion;

    // gender (numeric)
    uint32_t gender = 0;
    in.read(reinterpret_cast<char*>(&gender), sizeof(uint32_t));
    out_rec.gender = gender;

    // region: count + elements
    uint32_t region_count = 0;
    in.read(reinterpret_cast<char*>(&region_count), sizeof(uint32_t));
    out_rec.region.clear();
    for (uint32_t i = 0; i < region_count; ++i) {
        uint32_t pid = 0;
        in.read(reinterpret_cast<char*>(&pid), sizeof(uint32_t));
        out_rec.region.push_back(pid);
    }

    // age as uint32_t
    uint32_t age = 0;
    in.read(reinterpret_cast<char*>(&age), sizeof(uint32_t));
    out_rec.age = age;

    // clubs
    uint32_t clubs_count = 0;
    in.read(reinterpret_cast<char*>(&clubs_count), sizeof(uint32_t));
    out_rec.clubs.clear();
    for (uint32_t i = 0; i < clubs_count; ++i) {
        uint32_t cid = 0;
        in.read(reinterpret_cast<char*>(&cid), sizeof(uint32_t));
        out_rec.clubs.push_back(cid);
    }

    // token columns
    uint32_t cols_num = 0;
    in.read(reinterpret_cast<char*>(&cols_num), sizeof(uint32_t));
    out_rec.token_cols.clear();
    out_rec.token_cols.resize(cols_num);
    for (uint32_t ci = 0; ci < cols_num; ++ci) {
        uint32_t pairs_count = 0;
        in.read(reinterpret_cast<char*>(&pairs_count), sizeof(uint32_t));
        vector<pair<uint32_t,uint32_t>> vec;
        vec.reserve(pairs_count);
        for (uint32_t pi = 0; pi < pairs_count; ++pi) {
            uint32_t tid = 0;
            uint32_t cnt = 0;
            in.read(reinterpret_cast<char*>(&tid), sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(&cnt), sizeof(uint32_t));
            vec.push_back(make_pair(tid, cnt));
        }
        out_rec.token_cols[ci] = std::move(vec);
    }

    in.close();
    return true;
}
