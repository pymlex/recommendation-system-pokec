#include "tokenizer.h"
#include "lemmatizer_wrapper.h"
#include "vocab_builder.h"
#include "encoder.h"
#include "preprocess.h"
#include "graph_builder.h"
#include "hiercoarsener.h"
#include "recommender.h"
#include "utils.h"
#include "user_profile.h"
#include "data_explorer.h"
#include "column_stats.h"
#include "tfidf_index.h"
#include "evaluator.h"
#include <user_loader.h>
#include "eval.h"

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

    /*
    DataExplorer de;
    cout << "[main] running DataExplorer\n";
    de.analyze_users_encoded(users_encoded, adjacency_csv, textCols, "data/explore");
    */

    HierCoarsener hc(100, 0.5f);
    cout << "[main] HierCoarsener created (not used for non-coarsened run)\n";

    Recommender rec(&profiles_map, &adj_list);
    rec.set_field_normalizers(col_norms_map);
    rec.set_column_normalizers(col_norms_map);

    rec.compute_idf_from_profiles(textCols);
    cout << "[main] Recommender ready with precomputed idf for " << textCols.size() << " text columns\n";

    int test_uid = profiles_map.begin() != profiles_map.end() ? profiles_map.begin()->first : 1;
    cout << "Top profile-based (graph-registration) recommendations for user " << test_uid << ":\n";
    auto r = rec.recommend_graph_registration(test_uid, 10);
    for (size_t i = 0; i < r.size(); ++i)
        cout << "User " << r[i].first << " score=" << r[i].second << "\n";

    {
        cout << "[main] running evaluation sample tests (sample_size=10000, k=10)\n";
        Recommender rec_for_eval = rec; // copy small; fields are shallow maps
        EvalResult er = evaluate_recommender_sample(profiles_map, adj_list, rec_for_eval, textCols, 10000, 10);
        cout << "[main] EVAL (graph-registration) hit@10=" << er.hit_at_k
             << " prec@10=" << er.precision_at_k << " rec@10=" << er.recall_at_k << "\n";
    }

    return 0;
}
