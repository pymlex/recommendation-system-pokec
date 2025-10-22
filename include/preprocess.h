#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <string>
#include <vector>
#include "tokenizer.h"
using namespace std;
vector<string> split_tab(const string& line);
vector<vector<string>> preprocess_profiles(const string& path, Tokenizer& tok, size_t max_rows = 0);
void save_df_csv(const string& outpath, const vector<vector<string>>& df);
#endif
