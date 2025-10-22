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
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cout << "Usage: kurs <profiles_tsv> <relationships_txt>\n";
        return 1;
    }
    string profiles = argv[1];
    string rels = argv[2];

    /*
    unordered_map<string,int> club_to_id;
    vector<ClubInfo> id2club;
    extract_clubs(profiles, club_to_id, id2club);
    save_clubs_map("data/clubs_map.csv", id2club);
    */

    vector<string> textCols = {"languages", "hobbies", "music", "movies", "profession"};

    Tokenizer tok;
    Lemmatiser lemma("data/lem-me-sk.bin");

    VocabBuilder vb(textCols);
    vb.pass1(profiles, tok, lemma);
    vb.save_vocab("data/vocabs");

    Encoder enc(textCols, vb.token2id_per_col, vb.club_to_id);
    enc.pass2(profiles, "data/users_encoded.csv");

    vector<vector<string>> df = preprocess_profiles(profiles, tok, 10000);
    FeatureExtractor fe;
    fe.build_from_df(df);

    GraphBuilder gb;
    gb.load_edges(rels, 0);
    unordered_map<int, vector<int>> adj_list = build_adj_list(gb.adjacency);

    HierCoarsener hc(100, 0.5f);
    hc.coarsen(fe.user_tfidf, adj_list, 1);

    Recommender rec(&fe.user_tfidf, &adj_list);
    int test_uid = df.size() ? atoi(df[0][0].c_str()) : 1;

    cout << "Top recommendations for user " << test_uid << ":\n";
    vector<pair<int,float>> r = rec.recommend_by_cosine(test_uid, 10);
    for (size_t i = 0; i < r.size(); ++i)
        cout << "User " << r[i].first << " score=" << r[i].second << "\n";

    float hit = evaluate_holdout_hit_at_k(adj_list, fe.user_tfidf, 200, 10);
    cout << "Holdout hit@10 sample=200 -> " << hit << "\n";
    cout << "Supernodes: " << hc.super_features.size() << "\n";

    vector<pair<int,float>> sr = rec.recommend_from_supernodes(test_uid, hc.super_features, 5);
    cout << "Top supernodes for user " << test_uid << ":\n";
    for (size_t i = 0; i < sr.size(); ++i)
        cout << "Supernode " << sr[i].first << " score=" << sr[i].second << "\n";

    return 0;
}
