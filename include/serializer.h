#ifndef SERIALIZER_H
#define SERIALIZER_H
#include <string>
using namespace std;
bool csv_to_bin_index(const string& users_csv, const string& out_bin, const string& out_index, int num_token_cols);
#endif
