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
#include "clubs_extractor.h"

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

    // columns
    const string TEXT_COLS_PATH = "config/text_columns.txt";
    vector<string> textCols = load_text_columns_from_file(TEXT_COLS_PATH);
    cout << "[main] loaded " << textCols.size() << " text columns from " << TEXT_COLS_PATH << "\n";

    // clubs
    /*
    unordered_map<string,int> club_to_id;
    vector<ClubInfo> id2club;
    extract_clubs(profiles, club_to_id, id2club);
    save_clubs_map("data/clubs_map.csv", id2club);
    cout << "[main] extracted " << id2club.size() << " clubs\n";
    */

    // tokenizer / lemmatizer
    Tokenizer tok;
    Lemmatiser lemma("data/lem-me-sk.bin");

    // build vocabulary / token ids per column
    VocabBuilder vb(textCols);
    vb.pass1(profiles, tok, lemma);
    vb.save_vocab("data/vocabs");
    cout << "[main] vocab built for " << textCols.size() << " columns\n";

    // encode users CSV
    Encoder enc(textCols, vb.token2id_per_col, vb.club_to_id);
    enc.pass2(profiles, "data/users_encoded.csv");
    cout << "[main] users encoded -> data/users_encoded.csv\n";

    // adjacency: prefer CSV serialized version; if missing — build from raw relations file and save CSV
    GraphBuilder gb;
    const string adjacency_csv = "data/adjacency.csv";
    bool loaded = gb.load_serialized(adjacency_csv);
    if (! loaded) {
        gb.load_edges(rels, 0);
        gb.save_serialized(adjacency_csv);
    }
    cout << "[main] loaded adjacency from " << adjacency_csv << "\n";

    unordered_map<int, vector<int>> adj_list = build_adj_list(gb.adjacency);

    // preprocess a limited sample to build features for demonstration
    vector<vector<string>> df = preprocess_profiles(profiles, tok, 10000);
    cout << "[main] preprocessed " << df.size() << " rows (sample)\n";

    FeatureExtractor fe;
    fe.build_from_df(df);
    cout << "[main] feature extractor built (user_tfidf size = " << fe.user_tfidf.size() << ")\n";

    HierCoarsener hc(100, 0.5f);
    hc.coarsen(fe.user_tfidf, adj_list, 1);
    cout << "[main] coarsening done; supernodes = " << hc.super_features.size() << "\n";

    Recommender rec(&fe.user_tfidf, &adj_list);
    int test_uid = df.size() ? atoi(df[0][0].c_str()) : 1;

    cout << "Top recommendations for user " << test_uid << ":\n";
    vector<pair<int,float>> r = rec.recommend_by_cosine(test_uid, 10);
    for (size_t i = 0; i < r.size(); ++i)
        cout << "User " << r[i].first << " score=" << r[i].second << "\n";

    float hit = evaluate_holdout_hit_at_k(adj_list, fe.user_tfidf, 200, 10);
    cout << "Holdout hit@10 sample=200 -> " << hit << "\n";

    cout << "Top supernodes for user " << test_uid << ":\n";
    vector<pair<int,float>> sr = rec.recommend_from_supernodes(test_uid, hc.super_features, 5);
    for (size_t i = 0; i < sr.size(); ++i)
        cout << "Supernode " << sr[i].first << " score=" << sr[i].second << "\n";

    return 0;
}
