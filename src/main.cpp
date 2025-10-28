#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include "vocab_builder.h"
#include "encoder.h"
#include "preprocess.h"
#include "feature_extractor.h"
#include "graph_builder.h"
#include "hiercoarsener.h"
#include "recommender.h"
#include "utils.h"
#include "user_profile.h"
#include "data_explorer.h"
#include "column_stats.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
    const string profiles = "data/soc-pokec-profiles.txt";
    const string rels = "data/soc-pokec-relationships.txt";
    const string TEXT_COLS_PATH = "config/text_columns.txt";
    vector<string> textCols = load_text_columns_from_file(TEXT_COLS_PATH);

    Tokenizer tok;
    Lemmatiser lemma("data/lem-me-sk.bin");

    VocabBuilder vb(textCols);

    const string DATA_DIR = "data";
    bool loaded_vocab = vb.load_vocab(DATA_DIR);
    if (! loaded_vocab) {
        vb.pass1(profiles, tok, lemma);
        vb.save_vocab(DATA_DIR);
        cout << "[main] vocab built and saved to " << DATA_DIR << "\n";
    } else {
        cout << "[main] vocab loaded from " << DATA_DIR << "\n";
    }

    GraphBuilder gb;
    const string adjacency_csv = "data/adjacency.csv";
    bool loaded = gb.load_serialized(adjacency_csv);
    if (! loaded) {
        gb.load_edges(rels, 0);
        gb.save_serialized(adjacency_csv);
        cout << "[main] adjacency built and saved to " << adjacency_csv << "\n";
    } else {
        cout << "[main] adjacency loaded from " << adjacency_csv << "\n";
    }

    unordered_map<int, vector<int>> adj_list = build_adj_list(gb.adjacency);

    const string users_encoded = "data/users_encoded.csv";
    {
        ifstream f(users_encoded);
        bool exists = f.is_open();
        f.close();
        if (! exists) {
            Encoder enc(
                textCols,
                vb.token2id_per_col,
                vb.club_to_id,
                vb.address_part1_to_id,
                vb.address_part2_to_id,
                vb.address_part3_to_id,
                adj_list
            );
            enc.pass2(profiles, users_encoded);
            cout << "[main] users encoded and saved to " << users_encoded << "\n";
        } else {
            cout << "[main] encoded users found in " << users_encoded << "\n";
        }
    }

    unordered_map<int, UserProfile> profiles_map;
    bool ok = load_users_encoded(users_encoded, textCols, profiles_map);
    if (! ok) {
        cout << "[main] cannot load users_encoded.csv\n";
        return 1;
    }
    cout << "[main] loaded profiles: " << profiles_map.size() << "\n";

    const string median_path = DATA_DIR + "/median_age.txt";
    int median_age = 0;
    if (load_median_age(median_path, median_age)) {
        cout << "[main] loaded median_age=" << median_age << " from " << median_path << "\n";
    } else {
        cout << "[main] median not found, computing median from profiles\n";
        median_age = compute_median_age_from_profiles(profiles_map);
        if (median_age > 0) {
            save_median_age(median_path, median_age);
            cout << "[main] computed median_age=" << median_age << " and saved to " << median_path << "\n";
        } else {
            cout << "[main] computed median_age=0\n";
        }
    }

    int replaced = fill_missing_ages(profiles_map, median_age);
    cout << "[main] replaced " << replaced << " zero-ages with median_age=" << median_age << "\n";

    const string norms_path = DATA_DIR + "/column_normalizers.csv";
    unordered_map<string, pair<float,float>> col_norms_map;
    if (load_column_normalizers(norms_path, col_norms_map)) {
        cout << "[main] loaded column normalizers from " << norms_path << " (" << col_norms_map.size() << " entries)\n";
    } else {
        cout << "[main] column_normalizers.csv not found or invalid, computing normalizers (sample_size=100000, comps_per_user=5)\n";
        col_norms_map = compute_column_normalizers(profiles_map, textCols, 100000, 5);
        if (save_column_normalizers(norms_path, col_norms_map))
            cout << "[main] saved column normalizers to " << norms_path << " (" << col_norms_map.size() << " entries)\n";
        else
            cout << "[main] cannot save column normalizers to " << norms_path << "\n";
    }

    DataExplorer de;
    cout << "[main] running DataExplorer\n";
    de.analyze_users_encoded(users_encoded, adjacency_csv, textCols, "data/explore");

    FeatureExtractor fe;
    cout << "[main] running FeatureExtractor\n";
    vector<vector<string>> df = preprocess_profiles(profiles, tok, 10000);
    fe.build_from_df(df);
    cout << "[main] FeatureExtractor ready\n";

    HierCoarsener hc(100, 0.5f);
    cout << "[main] running HierCoarsener\n";
    unordered_map<int, unordered_map<int,float>> tfidf_copy = fe.user_tfidf;
    hc.coarsen(tfidf_copy, adj_list, 1);
    cout << "[main] HierCoarsener done\n";

    Recommender rec_profiles(&profiles_map, &adj_list);
    rec_profiles.set_field_normalizers(col_norms_map);

    int test_uid = df.size() ? atoi(df[0][0].c_str()) : (profiles_map.begin() != profiles_map.end() ? profiles_map.begin()->first : 1);

    cout << "Top profile-based recommendations for user " << test_uid << ":\n";
    vector<pair<int,float>> r = rec_profiles.recommend_by_profile(test_uid, 10);
    for (size_t i = 0; i < r.size(); ++i)
        cout << "User " << r[i].first << " score=" << r[i].second << "\n";

    Recommender rec_tfidf(&fe.user_tfidf, &adj_list);
    float hit = evaluate_holdout_hit_at_k(adj_list, fe.user_tfidf, 200, 10);
    cout << "Holdout hit@10 sample=200 -> " << hit << "\n";

    cout << "Top supernodes for user " << test_uid << ":\n";
    vector<pair<int,float>> sr = rec_tfidf.recommend_from_supernodes(test_uid, hc.super_features, 5);
    for (size_t i = 0; i < sr.size(); ++i)
        cout << "Supernode " << sr[i].first << " score=" << sr[i].second << "\n";

    return 0;
}
