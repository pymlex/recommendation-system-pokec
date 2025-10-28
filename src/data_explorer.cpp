#include "data_explorer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <filesystem>
#include <iomanip>

#ifdef USE_MATPLOT
#include <matplot/matplot.h>
#endif

using namespace std;

static vector<string> split_csv_line(const string& line)
{
    vector<string> out;
    string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        char c = line[i];
        if (c == '"') { in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static double mean_double(const vector<double>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double) v.size();
}

static double stddev_double(const vector<double>& v, double mu)
{
    if (v.size() < 2) return 0.0;
    double s = 0.0;
    for (double x : v) { double d = x - mu; s += d * d; }
    return sqrt(s / (double) (v.size() - 1));
}

static int median_int(vector<int> v)
{
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2) return v[n/2];
    return (v[n/2 - 1] + v[n/2]) / 2;
}

struct HeaderIdx {
    int user_id = -1;
    int public_idx = -1;
    int gender = -1;
    int region_id = -1;
    int age = -1;
    vector<int> text_token_indices;
};

static HeaderIdx parse_header(const vector<string>& header_cols, const vector<string>& text_columns)
{
    HeaderIdx hi;
    for (size_t i = 0; i < header_cols.size(); ++i) {
        string h = header_cols[i];
        if (h == "user_id") hi.user_id = (int)i;
        else if (h == "public") hi.public_idx = (int)i;
        else if (h == "gender") hi.gender = (int)i;
        else if (h == "region_id") hi.region_id = (int)i;
        else if (h == "age") hi.age = (int)i;
    }
    hi.text_token_indices.resize(text_columns.size(), -1);
    for (size_t t = 0; t < text_columns.size(); ++t) {
        string key = text_columns[t] + "_tokens";
        for (size_t i = 0; i < header_cols.size(); ++i) {
            if (header_cols[i] == key) { hi.text_token_indices[t] = (int)i; break; }
        }
    }
    return hi;
}

static void read_adjacency_degrees(const string& adjacency_csv, vector<int>& degs, int& total_edges)
{
    degs.clear();
    total_edges = 0;
    ifstream adjin(adjacency_csv);
    if (!adjin.is_open()) return;
    string aline;
    while (getline(adjin, aline)) {
        if (aline.empty()) continue;
        stringstream ss(aline);
        string token;
        vector<string> toks;
        while (getline(ss, token, ',')) toks.push_back(token);
        if (toks.size() == 0) continue;
        int deg = 0;
        for (size_t i = 1; i < toks.size(); ++i) if (!toks[i].empty()) ++deg;
        degs.push_back(deg);
        total_edges += deg;
    }
    adjin.close();
}

static void write_stats_file(const string& path,
                             size_t users_count,
                             double deg_mean, double deg_std, int deg_med,
                             double age_mean, double age_std, int age_med,
                             int gender_1, int gender_0,
                             int public_1, int public_0,
                             int total_edges)
{
    ofstream out(path);
    out << "users: " << users_count << "\n";
    out << "degree: mean=" << deg_mean << " std=" << deg_std << " median=" << deg_med << "\n";
    out << "age: mean=" << age_mean << " std=" << age_std << " median=" << age_med << "\n";
    out << "gender: 1=" << gender_1 << " 0=" << gender_0 << "\n";
    out << "public: 1=" << public_1 << " 0=" << public_0 << "\n";
    out << "total edges: " << total_edges << "\n";
    out.close();
}

#ifdef USE_MATPLOT
static void plot_histogram_matplot(const vector<int>& data, const string& stitle, const string& sxlabel, const string& outpath)
{
    if (data.empty()) return;
    vector<double> vec;
    vec.reserve(data.size());
    for (int v : data) vec.push_back((double)v);
    matplot::figure();
    matplot::hist(vec);
    matplot::title(stitle);
    matplot::xlabel(sxlabel);
    matplot::ylabel("count");
    matplot::save(outpath);
}

static void plot_bar_counts_matplot(const vector<pair<string,int>>& items, const string& stitle, const string& outpath)
{
    if (items.empty()) return;
    vector<string> labels;
    vector<double> values;
    for (const auto &p : items) { labels.push_back(p.first); values.push_back((double)p.second); }
    matplot::figure();
    matplot::bar(values);
    matplot::title(stitle);
    matplot::save(outpath);
}
#endif

void DataExplorer::analyze_users_encoded(const string& users_encoded_csv,
                                         const string& adjacency_csv,
                                         const vector<string>& text_columns,
                                         const string& out_prefix)
{
    filesystem::create_directories(out_prefix);

    ifstream in(users_encoded_csv);
    if (! in.is_open()) return;

    string header;
    if (!getline(in, header)) return;
    vector<string> header_cols = split_csv_line(header);
    HeaderIdx hi = parse_header(header_cols, text_columns);

    string line;

    vector<int> ages_nonzero;
    unordered_map<int,int> addr_count;
    int gender_1 = 0, gender_0 = 0;
    int public_1 = 0, public_0 = 0;
    vector<int> null_counts(text_columns.size(), 0);
    size_t users_count = 0;

    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> parts = split_csv_line(line);
        if (parts.size() == 0) continue;
        int uid = 0;
        if (hi.user_id >= 0 && hi.user_id < (int)parts.size()) uid = atoi(parts[hi.user_id].c_str());
        ++users_count;

        string age_s = (hi.age >= 0 && hi.age < (int)parts.size()) ? parts[hi.age] : "";
        int age = 0;
        if (!age_s.empty()) age = atoi(age_s.c_str());
        if (age > 0) ages_nonzero.push_back(age);

        string gender_s = (hi.gender >= 0 && hi.gender < (int)parts.size()) ? parts[hi.gender] : "";
        if (gender_s == "1") ++gender_1; else ++gender_0;

        string public_s = (hi.public_idx >= 0 && hi.public_idx < (int)parts.size()) ? parts[hi.public_idx] : "";
        if (public_s == "1") ++public_1; else ++public_0;

        string region_field = (hi.region_id >= 0 && hi.region_id < (int)parts.size()) ? parts[hi.region_id] : "";
        if (!region_field.empty()) {
            int aid = atoi(region_field.c_str());
            addr_count[aid] += 1;
        }

        for (size_t t = 0; t < text_columns.size(); ++t) {
            int idx = hi.text_token_indices[t];
            string cell = (idx >= 0 && idx < (int)parts.size()) ? parts[idx] : "";
            if (cell.empty()) ++null_counts[t];
        }
    }
    in.close();

    vector<int> degs;
    int total_edges = 0;
    read_adjacency_degrees(adjacency_csv, degs, total_edges);

    double deg_mean = 0.0, deg_std = 0.0;
    int deg_med = 0;
    if (!degs.empty()) {
        vector<double> dd;
        dd.reserve(degs.size());
        for (int x : degs) dd.push_back((double)x);
        deg_mean = mean_double(dd);
        deg_std = stddev_double(dd, deg_mean);
        deg_med = median_int(degs);
    }

    double age_mean = 0.0, age_std = 0.0;
    int age_med = 0;
    if (!ages_nonzero.empty()) {
        vector<double> ad;
        ad.reserve(ages_nonzero.size());
        for (int x : ages_nonzero) ad.push_back((double)x);
        age_mean = mean_double(ad);
        age_std = stddev_double(ad, age_mean);
        age_med = median_int(ages_nonzero);
    }

    write_stats_file(out_prefix + "/explore_stats.txt",
                     users_count,
                     deg_mean, deg_std, deg_med,
                     age_mean, age_std, age_med,
                     gender_1, gender_0,
                     public_1, public_0,
                     total_edges);

    {
        ofstream out_deg_csv(out_prefix + "/degree_hist.csv");
        for (size_t i = 0; i < degs.size(); ++i) out_deg_csv << degs[i] << "\n";
    }
    {
        ofstream out_age_csv(out_prefix + "/ages.csv");
        for (size_t i = 0; i < ages_nonzero.size(); ++i) out_age_csv << ages_nonzero[i] << "\n";
    }
    {
        ofstream out_addr_csv(out_prefix + "/addr_counts.csv");
        vector<pair<int,int>> addr_vec;
        addr_vec.reserve(addr_count.size());
        for (auto &p : addr_count) addr_vec.push_back(p);
        sort(addr_vec.begin(), addr_vec.end(), [](const pair<int,int>& A, const pair<int,int>& B){ return A.second > B.second; });
        for (auto &p : addr_vec) out_addr_csv << p.first << "," << p.second << "\n";
    }
    {
        ofstream out_nulls(out_prefix + "/nulls_per_textcol.csv");
        for (size_t i = 0; i < text_columns.size(); ++i)
            out_nulls << text_columns[i] << "," << null_counts[i] << "\n";
    }
    {
        ofstream out_gender_public(out_prefix + "/gender_public.csv");
        out_gender_public << "gender_1," << gender_1 << "\n";
        out_gender_public << "gender_0," << gender_0 << "\n";
        out_gender_public << "public_1," << public_1 << "\n";
        out_gender_public << "public_0," << public_0 << "\n";
    }

#ifdef USE_MATPLOT
    try {
        vector<pair<string,int>> top_addrs;
        {
            vector<pair<int,int>> addr_vec;
            addr_vec.reserve(addr_count.size());
            for (auto &p : addr_count) addr_vec.push_back(p);
            sort(addr_vec.begin(), addr_vec.end(), [](const pair<int,int>& A, const pair<int,int>& B){ return A.second > B.second; });
            size_t take = addr_vec.size() < 30 ? addr_vec.size() : 30;
            for (size_t i = 0; i < take; ++i) top_addrs.push_back(make_pair(to_string(addr_vec[i].first), addr_vec[i].second));
        }
        plot_histogram_matplot(degs, "Degree distribution", "degree", out_prefix + "/degree_hist.png");
        plot_histogram_matplot(vector<int>(ages_nonzero.begin(), ages_nonzero.end()), "Age distribution (non-zero)", "age", out_prefix + "/age_hist.png");
        plot_bar_counts_matplot(top_addrs, "Top addresses (by id)", out_prefix + "/top_addresses.png");
        vector<pair<string,int>> null_items;
        for (size_t i = 0; i < text_columns.size(); ++i) null_items.push_back(make_pair(text_columns[i], null_counts[i]));
        sort(null_items.begin(), null_items.end(), [](const pair<string,int>& A, const pair<string,int>& B){ return A.second > B.second; });
        if (null_items.size() > 40) null_items.resize(40);
        plot_bar_counts_matplot(null_items, "Nulls per text column (top 40)", out_prefix + "/nulls_per_textcol.png");
        vector<pair<string,int>> gp = { {"gender_1", gender_1}, {"gender_0", gender_0}, {"public_1", public_1}, {"public_0", public_0} };
        plot_bar_counts_matplot(gp, "Gender / Public distribution", out_prefix + "/gender_public.png");
    } catch (...) {}
#endif
}
