#include "clubs_extractor.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>


using namespace std;


static string slug_normalize(const string& s) {
    string out;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        unsigned char uc = (unsigned char) c;
        if ( (uc >= '0' && uc <= '9') || (uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z') || c == '-' ) {
            if (uc >= 'A' && uc <= 'Z') 
                out.push_back((char)(uc + 32));
            else 
                out.push_back(c);
        } else {
            if (out.empty() == false && out.back() != '-') 
                out.push_back('-');
        }
    }

    while (out.size() > 0 && out.back() == '-') 
        out.pop_back();

    return out;
}

void extract_clubs(const string& profiles_tsv, 
                   unordered_map<string,int>& club_to_id, 
                   vector<ClubInfo>& id2club) {
    ifstream in(profiles_tsv);
    string line;
    regex href_re("<a[^>]*href=\"/klub/([^\"]+)\"[^>]*>([^<]*)</a>");

    int next_id = 0;
    while (getline(in, line)) {
        if (line.empty()) 
            continue;

        string::const_iterator start = line.cbegin();
        smatch m;
        while (regex_search(start, line.cend(), m, href_re)) {
            string raw_slug = m[1].str();
            string raw_title = m[2].str();
            string slug = slug_normalize(raw_slug);
            string title = raw_title;
            if (slug.empty() && title.empty()) { 
                start = m.suffix().first; 
                continue; 
            }
            if (club_to_id.find(slug) == club_to_id.end()) {
                club_to_id[slug] = next_id;
                ClubInfo info;
                info.id = next_id;
                info.slug = slug;
                info.title = title;
                id2club.push_back(info);
                ++next_id;
            }
            start = m.suffix().first;
        }
    }
}

void save_clubs_map(const string& out_csv, const vector<ClubInfo>& id2club) {
    ofstream out(out_csv);
    out << "club_id,slug,title\n";
    for (size_t i = 0; i < id2club.size(); ++i) {
        const ClubInfo& c = id2club[i];
        bool need_quote = (c.title.find(',') != string::npos || c.title.find('"') != string::npos);
        out << c.id << "," << c.slug << ",";

        if (need_quote) 
            out << '"';

        for (size_t j = 0; j < c.title.size(); ++j) {
            char ch = c.title[j];
            if (ch == '"') 
                out << "\"\""; 
            else 
            out << ch;
        }

        if (need_quote) 
            out << '"';
            
        out << "\n";
    }
}
