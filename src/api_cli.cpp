#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>

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
#include "recommendation_tests.h"
#include "user_loader.h"
#include "ui.h"

using namespace std;

static string json_escape(const string &s) {
    string out;
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (int)c);
                    out += buf;
                } else out += c;
        }
    }
    return out;
}

static void write_profile_json(const UserProfile &p, ostream &os) {
    os << "{";
    os << "\"user_id\":" << p.user_id << ",";
    os << "\"public_flag\":" << p.public_flag << ",";
    os << "\"completion_percentage\":" << p.completion_percentage << ",";
    os << "\"gender\":" << p.gender << ",";
    os << "\"age\":" << p.age << ",";
    os << "\"region_parts\":[" << p.region_parts[0] << "," << p.region_parts[1] << "," << p.region_parts[2] << "],";
    os << "\"clubs\":[";
    for (size_t i = 0; i < p.clubs.size(); ++i) {
        if (i) os << ",";
        os << p.clubs[i];
    }
    os << "],";
    os << "\"friends\":[";
    for (size_t i = 0; i < p.friends.size(); ++i) {
        if (i) os << ",";
        os << p.friends[i];
    }
    os << "],";
    os << "\"token_cols\":[";
    for (size_t t = 0; t < p.token_cols.size(); ++t) {
        if (t) os << ",";
        os << "{";
        bool first = true;
        for (auto &pr : p.token_cols[t]) {
            if (!first) os << ",";
            first = false;
            os << "\"" << pr.first << "\":" << pr.second;
        }
        os << "}";
    }
    os << "]";
    os << "}";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(true);
    cin.tie(nullptr);

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
        cerr << "[api_cli] vocab built and saved to " << DATA_DIR << "\n";
    } else {
        cerr << "[api_cli] vocab loaded from " << DATA_DIR << "\n";
    }

    GraphBuilder gb;
    const string adjacency_csv = "data/adjacency.csv";
    bool loaded = gb.load_serialized(adjacency_csv);
    if (! loaded) {
        gb.load_edges(rels, 0);
        gb.save_serialized(adjacency_csv);
        cerr << "[api_cli] adjacency built and saved to " << adjacency_csv << "\n";
    } else {
        cerr << "[api_cli] adjacency loaded from " << adjacency_csv << "\n";
    }

    unordered_map<int, vector<int>> adj_list = build_adj_list(gb.adjacency);

    const string users_encoded = "data/users_encoded.csv";

    unordered_map<int, UserProfile> profiles_map;

    size_t to_load = 0;
    if (argc > 1) {
        try { to_load = (size_t)stoi(argv[1]); } catch(...) { to_load = 0; }
    }

    bool ok = load_users_encoded(users_encoded, textCols, profiles_map, to_load);
    if (! ok) {
        cerr << "[api_cli] cannot load users_encoded.csv\n";
        return 1;
    }
    cerr << "[api_cli] loaded profiles: " << profiles_map.size() << "\n";

    int median_age = 0;
    const string median_path = DATA_DIR + "/median_age.txt";
    if (load_median_age(median_path, median_age)) {
        cerr << "[api_cli] loaded median_age=" << median_age << " from " << median_path << "\n";
    } else {
        median_age = compute_median_age_from_profiles(profiles_map);
        if (median_age > 0) {
            save_median_age(median_path, median_age);
            cerr << "[api_cli] computed median_age=" << median_age << " and saved to " << median_path << "\n";
        } else {
            cerr << "[api_cli] computed median_age=0\n";
        }
    }
    int replaced = fill_missing_ages(profiles_map, median_age);
    cerr << "[api_cli] replaced " << replaced << " zero-ages with median_age=" << median_age << "\n";

    unordered_map<string, pair<float,float>> col_norms_map;
    const string norms_path = DATA_DIR + "/column_normalizers.csv";
    if (load_column_normalizers(norms_path, col_norms_map)) {
        cerr << "[api_cli] loaded column normalizers from " << norms_path << " (" << col_norms_map.size() << " entries)\n";
    } else {
        cerr << "[api_cli] column_normalizers.csv not found or invalid\n";
    }

    Recommender rec(&profiles_map, &adj_list);
    rec.set_field_normalizers(col_norms_map);
    rec.set_column_normalizers(col_norms_map);
    rec.compute_idf_from_profiles(textCols);
    rec.set_text_columns(textCols);

    unordered_map<int,string> club_id_to_name;
    for (auto &kv : vb.club_to_id) club_id_to_name[kv.second] = kv.first;

    cout << "READY" << endl;
    cout.flush();

    string line;
    while (true) {
        if (! std::getline(cin, line)) break;
        if (line.size() == 0) {
            cout << "{}" << endl;
            cout.flush();
            continue;
        }
        string cmd;
        int uid = -1;
        {
            istringstream iss(line);
            iss >> cmd;
            if (cmd == "USER") iss >> uid;
        }
        if (cmd == "PING") {
            cout << "{\"ok\":true}" << endl;
            cout.flush();
            continue;
        }
        if (cmd == "EXIT") {
            cout << "{\"ok\":true, \"exiting\":true}" << endl;
            cout.flush();
            break;
        }
        if (cmd == "USER" && uid >= 0) {
            auto it = profiles_map.find(uid);
            if (it == profiles_map.end()) {
                cout << "{\"error\":\"not found\",\"user_id\":" << uid << "}" << endl;
                cout.flush();
                continue;
            }
            ostringstream os;
            os << "{";
            os << "\"profile\":";
            write_profile_json(it->second, os);
            os << ",";
            os << "\"recommendations\":{";
            auto out_g = rec.recommend_graph_registration(uid, 20, 5000);
            os << "\"graph\":[";
            for (size_t i = 0; i < out_g.size(); ++i) {
                if (i) os << ",";
                os << "{\"id\":" << out_g[i].first << ",\"score\":" << std::fixed << std::setprecision(6) << out_g[i].second << "}";
            }
            os << "],";
            auto out_c = rec.recommend_collaborative(uid, 20, 5000);
            os << "\"collaborative\":[";
            for (size_t i = 0; i < out_c.size(); ++i) {
                if (i) os << ",";
                os << "{\"id\":" << out_c[i].first << ",\"score\":" << std::fixed << std::setprecision(6) << out_c[i].second << "}";
            }
            os << "],";
            auto out_i = rec.recommend_by_interest(uid, 20, 5000);
            os << "\"interest\":[";
            for (size_t i = 0; i < out_i.size(); ++i) {
                if (i) os << ",";
                os << "{\"id\":" << out_i[i].first << ",\"score\":" << std::fixed << std::setprecision(6) << out_i[i].second << "}";
            }
            os << "],";
            auto out_cl = rec.recommend_clubs_collab(uid, 20, 5000);
            os << "\"clubs\":[";
            for (size_t i = 0; i < out_cl.size(); ++i) {
                if (i) os << ",";
                int cid = out_cl[i].first;
                os << "{\"id\":" << cid << ",\"score\":" << std::fixed << std::setprecision(6) << out_cl[i].second;
                auto itn = club_id_to_name.find(cid);
                if (itn != club_id_to_name.end()) {
                    os << ",\"name\":\"" << json_escape(itn->second) << "\"";
                }
                os << "}";
            }
            os << "]";
            os << "}";
            os << "}";
            cout << os.str() << endl;
            cout.flush();
            continue;
        }
        cout << "{\"error\":\"unknown command\"}" << endl;
        cout.flush();
    }

    return 0;
}
