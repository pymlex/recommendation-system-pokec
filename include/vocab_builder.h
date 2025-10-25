#ifndef VOCAB_BUILDER_H
#define VOCAB_BUILDER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "tokenizer.h"
#include "lemmatizer_wrapper.h"

using namespace std;

struct VocabBuilder {
    explicit VocabBuilder(const vector<string> &colKeys);

    // Проход по профилям для построения словарей
    void pass1(const string &profiles_tsv, Tokenizer &tok, Lemmatiser &lem);

    // Сохранить словари и карты в указанную папку (обычно "data")
    // Записывает:
    //   - <out_dir>/tokens.csv           (columns: column,token,term_id,docfreq)
    //   - <out_dir>/clubs_map.csv        (club_id,slug,title)
    //   - <out_dir>/addresses_map.csv    (address_id,address)
    void save_vocab(const string &out_dir) const;

    // Загрузить словари и карты из указанной папки
    // Возвращает true при успешной загрузке всех необходимых файлов.
    bool load_vocab(const string &in_dir);

    unordered_map<string, unordered_map<string,int>> token2id_per_col;
    unordered_map<string, unordered_map<int,int>> docfreq_per_col;
    unordered_map<string,int> club_to_id;
    unordered_map<string,string> club_slug_to_title;
    unordered_map<string,int> address_to_id;

private:
    vector<string> colKeys;

    void process_line_clubs(const string& line);
    void process_line_tokens(const vector<string>& cols, Tokenizer &tok, Lemmatiser &lem);
    static string normalize_slug(const string& raw);
    static string normalize_address(const string& raw);

    static string csv_escape(const string& s);
    static vector<string> split_csv_line(const string& line);
};

#endif
