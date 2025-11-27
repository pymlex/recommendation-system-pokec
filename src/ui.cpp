#include "ui.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdlib> 

using namespace std;

static void clear_screen() {
    system("cls");
}

static void print_title() {
    //system("chcp 65001");
    cout << "\n";
    cout << "██████╗  ██████╗ ██╗  ██╗███████╗ ██████╗\n";
    cout << "██╔══██╗██╔═══██╗██║ ██╔╝██╔════╝██╔════╝\n";
    cout << "██████╔╝██║   ██║█████╔╝ █████╗  ██║     \n";
    cout << "██╔═══╝ ██║   ██║██╔═██╗ ██╔══╝  ██║     \n";
    cout << "██║     ╚██████╔╝██║  ██╗███████╗╚██████╗\n";
    cout << "╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝\n";
    cout << "              Interactive recommender\n\n";
}

static int menu_choose_numbered(const vector<string>& items) {
    while (true) {
        for (size_t i = 0; i < items.size(); ++i) {
            cout << "  [" << (i+1) << "] " << items[i] << "\n";
        }
        cout << "  [0] Cancel / Back\n";
        cout << "\nEnter choice number: ";
        int choice;
        if (!(cin >> choice)) {
            cin.clear();
            string dummy;
            getline(cin, dummy);
            cout << "Invalid input. Please enter a number.\n";
            continue;
        }
        if (choice < 0 || choice > (int)items.size()) {
            cout << "Choice out of range. Try again.\n";
            continue;
        }
        return choice;
    }
}

static string format_user_brief(const UserProfile& p) {
    string s = "id=" + to_string(p.user_id);
    if (p.age > 0) s += " age=" + to_string(p.age);
    if (p.gender >= 0) s += " gender=" + to_string(p.gender);
    s += " clubs=" + to_string(p.clubs.size());
    s += " friends=" + to_string(p.friends.size());
    return s;
}

void run_terminal_ui(unordered_map<int, UserProfile>& profiles,
                     const unordered_map<int, vector<int>>& adj_list,
                     Recommender& rec,
                     const unordered_map<int, string>& club_id_to_name,
                     const vector<string>& text_columns,
                     size_t loaded_users)
{
    print_title();
    while (true) {
        cout << "Loaded users: " << loaded_users << "\n\n";
        cout << "Enter user id to inspect (or 0 to exit): ";
        int uid;
        if (!(cin >> uid)) {
            cin.clear();
            string tmp; getline(cin, tmp);
            continue;
        }
        if (uid == 0) return;
        auto it = profiles.find(uid);
        if (it == profiles.end()) {
            cout << "User not found. Press Enter to continue.";
            string tmp; getline(cin, tmp); getline(cin, tmp);
            continue;
        }
        const UserProfile &p = it->second;
        while (true) {
            cout << "User overview:\n\n";
            cout << "  " << format_user_brief(p) << "\n\n";
            cout << "  Clubs (" << p.clubs.size() << "):\n";
            for (size_t i = 0; i < p.clubs.size() && i < 10; ++i) {
                int cid = (int)p.clubs[i];
                auto itn = club_id_to_name.find(cid);
                cout << "    " << cid << " : " << (itn != club_id_to_name.end() ? itn->second : string("<name?>")) << "\n";
            }
            cout << "\n  Friends (" << p.friends.size() << "):\n";
            for (size_t i = 0; i < p.friends.size() && i < 20; ++i) cout << "    " << p.friends[i] << (i+1<p.friends.size() ? "," : "") << "\n";
            cout << "\nChoose action:\n";
            vector<string> actions = {
                "Recommend friends (graph + friends-of-friends)",
                "Recommend friends (collaborative)",
                "Recommend friends by interest (profile similarity)",
                "Recommend clubs (collaborative)",
                "Back to user id input"
            };
            int choice = menu_choose_numbered(actions);
            if (choice == 0 || choice == (int)actions.size()) break;
            if (choice == 1) {
                cout << "Graph-based recommendations for user " << uid << ":\n\n";
                auto out = rec.recommend_graph_registration(uid, 20);
                for (auto &pr : out) cout << "  user " << pr.first << " score=" << pr.second << "\n";
                cout << "\nPress Enter to continue.";
                string tmp; getline(cin, tmp); getline(cin, tmp);
            } else if (choice == 2) {
                cout << "Collaborative recommendations for user " << uid << ":\n\n";
                auto out = rec.recommend_collaborative(uid, 20);
                for (auto &pr : out) cout << "  user " << pr.first << " score=" << pr.second << "\n";
                cout << "\nPress Enter to continue.";
                string tmp; getline(cin, tmp); getline(cin, tmp);
            } else if (choice == 3) {
                cout << "Interest-based recommendations for user " << uid << ":\n\n";
                auto out = rec.recommend_by_interest(uid, 20);
                for (auto &pr : out) cout << "  user " << pr.first << " score=" << pr.second << "\n";
                cout << "\nPress Enter to continue.";
                string tmp; getline(cin, tmp); getline(cin, tmp);
            } else if (choice == 4) {
                cout << "Club recommendations for user " << uid << ":\n\n";
                auto out = rec.recommend_clubs_collab(uid, 20);
                for (auto &pr : out) {
                    int cid = pr.first;
                    auto itn = club_id_to_name.find(cid);
                    cout << "  club " << cid << " score=" << pr.second << " name=" << (itn!=club_id_to_name.end() ? itn->second : string("<name?>")) << "\n";
                }
                cout << "\nPress Enter to continue.";
                string tmp; getline(cin, tmp); getline(cin, tmp);
            }
        }
    }
}
