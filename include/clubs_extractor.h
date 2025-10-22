#ifndef CLUBS_EXTRACTOR_H
#define CLUBS_EXTRACTOR_H
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct ClubInfo {
    int id;
    string slug;
    string title;
};
void extract_clubs(const string& profiles_tsv, unordered_map<string,int>& club_to_id, vector<ClubInfo>& id2club);
void save_clubs_map(const string& out_csv, const vector<ClubInfo>& id2club);
#endif
