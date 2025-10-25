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

    // data dir — файлы будут: data/tokens.csv, data/clubs_map.csv, data/addresses_map.csv
    const string DATA_DIR = "data";
    bool loaded_vocab = vb.load_vocab(DATA_DIR);
    if (! loaded_vocab) {
        // если не удалось загрузить — построим и сохраним
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
    }

    unordered_map<int, vector<int>> adj_list = build_adj_list(gb.adjacency);

    const string users_encoded = "data/users_encoded.csv";
    {
        ifstream f(users_encoded);
        bool exists = f.is_open();
        f.close();
        if (! exists) {
            Encoder enc(textCols, vb.token2id_per_col, vb.club_to_id, vb.address_to_id, adj_list);
            enc.pass2(profiles, users_encoded);
        }
    }

    unordered_map<int, UserProfile> profiles_map;
    int median_age = 0;
    bool ok = load_users_encoded(users_encoded, textCols, profiles_map, median_age);
    if (! ok) {
        cout << "[main] cannot load users_encoded.csv\n";
        return 1;
    }
    cout << "[main] loaded profiles: " << profiles_map.size() << " median_age=" << median_age << "\n";

    DataExplorer de;
    de.analyze_users_encoded(users_encoded, adjacency_csv, textCols, "data/explore");

    FeatureExtractor fe;
    vector<vector<string>> df = preprocess_profiles(profiles, tok, 10000);
    fe.build_from_df(df);

    HierCoarsener hc(100, 0.5f);
    unordered_map<int, unordered_map<int,float>> tfidf_copy = fe.user_tfidf;
    hc.coarsen(tfidf_copy, adj_list, 1);

    Recommender rec_profiles(&profiles_map, &adj_list);
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