#ifndef DATA_EXPLORER_H
#define DATA_EXPLORER_H

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

struct DataExplorer {
    void analyze_users_encoded(const string& users_encoded_csv,
                               const string& adjacency_csv,
                               const vector<string>& text_columns,
                               const string& out_prefix);
};

#endif