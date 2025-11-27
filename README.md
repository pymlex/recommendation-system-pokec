# Recommendation System — Pokec (C++)

## Overview

A C++ heuristic recommendation system built for the Slovak social network **Pokec** (≈1.6M users).
This project implements a memory-efficient, fast pipeline combining graph-based heuristics, collaborative signals and content (text) similarity (TF–IDF) to recommend friends and clubs. The core idea is pragmatic: keep encoded user profiles and a compact graph in memory, compute TF–IDF weights (IDF precomputed) and produce recommendations by scanning only the local neighborhood (friends + friends-of-friends) rather than the whole user base.

Key features

* Friend recommendation from partial user registration (friends + friends-of-friends).
* Collaborative friend recommendation using friend → friend-of-friend propagation weighted by similarities.
* Interest-based friend recommendation (profile similarity using TF–IDF for text columns + other structured fields).
* Collaborative club (subscription) recommendations.
* Fill-Aware Similarity (FAS) — similarity measure that accounts both for per-field similarity and how many fields are actually filled in (profile completion awareness).
* Lightweight interactive terminal UI for inspection and ad-hoc recommendations.
* Evaluation / offline testing routines (sample-based holdout testing, hit@k / precision / recall, club precision/recall).

---

## Project structure

```
.
├── .kaggle
├── .vscode
├── build
├── config
├── data
│   └── explore
├── include
│   ├── bin_reader.h
│   ├── column_stats.h
│   ├── data_explorer.h
│   ├── encoder.h
│   ├── eval.h
│   ├── evaluator.h
│   ├── graph_builder.h
│   ├── hiercoarsener.h
│   ├── lemmatizer_wrapper.h
│   ├── preprocess.h
│   ├── recommendation_tests.h
│   ├── recommender.h
│   ├── recommender_clubs.h
│   ├── recommender_graph.h
│   ├── recommender_similarity.h
│   ├── serializer.h
│   ├── tfidf_index.h
│   ├── tokenizer.h
│   ├── ui.h
│   ├── user_loader.h
│   ├── user_profile.h
│   ├── utils.h
│   └── vocab_builder.h
├── src
│   ├── bin_reader.cpp
│   ├── column_stats.cpp
│   ├── data_explorer.cpp
│   ├── encoder.cpp
│   ├── eval.cpp
│   ├── evaluator.cpp
│   ├── graph_builder.cpp
│   ├── hiercoarsener.cpp
│   ├── lemmatizer_wrapper.cpp
│   ├── main.cpp
│   ├── preprocess.cpp
│   ├── recommendation_tests.cpp
│   ├── recommender.cpp
│   ├── recommender_clubs.cpp
│   ├── recommender_graph.cpp
│   ├── recommender_similarity.cpp
│   ├── serializer.cpp
│   ├── tfidf_index.cpp
│   ├── tokenizer.cpp
│   ├── ui.cpp
│   ├── user_loader.cpp
│   ├── user_profile.cpp
│   ├── utils.cpp
│   └── vocab_builder.cpp
└── third_party
    ├── lemmagen  (modified lemmagen C++ with slovak dict)
    └── matplotplusplus
```

Notes:

* `third_party/lemmagen` contains a modified native C++ lemmatizer used with `data/lem-me-sk.bin` (Slovak dictionary).
* `data/` is the place for Pokec datasets and precomputed artifacts (adjacency CSV, encoded users, column normalizers, median_age, etc.).
* `config/text_columns.txt` lists text columns used for TF–IDF processing.

---

## High-level execution sequence (`main`)

1. **Read config** — load list of text columns from `config/text_columns.txt`.
2. **Tokenizer & lemmatizer** initialisation (`Tokenizer`, `Lemmatiser` using `data/lem-me-sk.bin`).
3. **Vocabulary** (`VocabBuilder`) load or build:

   * If vocabulary not found in `data/`, perform pass1 over raw profiles to build token vocabularies and club/address mappings and save.
4. **Graph** (`GraphBuilder`) load or build:

   * Load serialized adjacency from `data/adjacency.csv` if present; otherwise parse raw edges and serialize.
   * Convert graph into adjacency list `adj_list`.
5. **Encode users**:

   * If `data/users_encoded.csv` missing, run the `Encoder` to produce it (token ids, counts, club ids, friends lists).
6. **Load users**:

   * `load_users_encoded(...)` reads `users_encoded.csv` and fills `profiles_map`. The program asks at startup how many users to load (e.g. `100000` or `0` = all).
7. **Data cleanup**:

   * Load or compute `median_age` and replace missing ages.
8. **Column normalizers**:

   * Load `data/column_normalizers.csv` or compute column normalizers (sample-based).
9. **Recommender init**:

   * Create `Recommender` with pointers to `profiles_map` and `adj_list`.
   * Set field and column normalizers.
   * Compute IDF maps in the recommender (`compute_idf_from_profiles`) — IDF is precomputed and stored per text column for on-the-fly TF–IDF.
   * Set internal `text_columns` list.
10. **Interactive UI**:

    * Launch the terminal UI (`run_terminal_ui`) where user can input an ID, view profile summary, and pick recommendation options (graph / collaborative / interest / clubs).
11. **Testing / evaluation**:

    * Offline evaluators and test harnesses exist (sample-based) to produce hits / precision / recall metrics. The project contains `recommendation_tests` helpers used from `main` in batch mode.

---

## Recommendation functions — design & behavior

The recommender exposes several main methods (short descriptions):

### 1. `recommend_graph_registration(user, topk, candidate_limit)`

* Use when a user has only partial profile / has only listed a subset of friends.
* Candidate pool: the user's friends plus friends-of-friends (capped by `candidate_limit`).
* For each candidate (excluding existing friends and self), compute `FAS` between query profile and candidate profile; rank by FAS and return top `k`.

### 2. `recommend_collaborative(user, topk, candidate_limit)`

* Collaborative propagation using social graph neighborhood only.
* Compute similarity `sim(u, f)` between user and each direct friend `f`.
* For each friend-of-friend candidate `c`, compute score `score(c) = Σ_f sim(u,f) * sim(f,c)` where `sim(f,c)` is similarity between friend and candidate.
* Rank by score and return top `k`.
* Efficient: only traverses friends and friends-of-friends instead of global scan.

### 3. `recommend_by_interest(user, topk, candidate_limit)`

* Interest / content-based: find candidates among friends-of-friends with maximal FAS to user (prioritises profile content similarity).
* Uses TF–IDF cosine on text columns plus structured fields (age, region, clubs, friends, completion, gender, public) combined in FAS.

### 4. `recommend_clubs_collab(user, topk, candidate_limit)`

* Predict club subscriptions (which clubs to recommend) using collaborative signals:

  * Direct friends contribute club votes weighted by `sim(u, friend)`.
  * Friends-of-friends contribute proportionally to `sim(u, f) * sim(f, fof)`.
  * Clubs the user already has are excluded from recommendations.
* Output: ranked list of club ids and scores.

---

## Similarity: Fill-Aware Similarity (FAS)

FAS is the central similarity used by the system. It combines:

1. **Per-field similarity (S)** — for each field that is *present in both profiles* (non-empty), compute a local similarity `s_i`:

   * For text columns: TF–IDF cosine per column, optionally standardized using column normalizers.
   * For sets (clubs, friends): set overlap normalized via vector-set similarity.
   * For region: hierarchical comparison via region parts.
   * For numeric/range fields (age, completion): ratio-based similarity (min/max).
   * Each raw `s_i` is converted to a bounded similarity via a sigmoid of a z-score (z-score computed using field normalizers if available; fallback mapping otherwise).
2. **Aggregate S = average(sigmoid(z_i))** across all *common non-empty* fields.
3. **Fill factor F = (# of common non-empty fields) / (total possible fields)** where total possible fields = fixed fields (public, gender, completion, age, region, clubs, friends) + number_of_text_columns.
4. **FAS formula**:

$$\text{FAS} = \frac{2 \cdot S \cdot F}{S + F}$$

   * This harmonic-like combination ensures that similarity is high only if both the per-field similarity S is high *and* there is a substantial overlap in which fields are present (F). It penalizes high raw similarity computed on very few overlapping fields.

Usage notes:

* IDF for text columns is precomputed once and stored in the recommender (`idf_per_col`) so TF–IDF weighting during similarity is efficient.
* Column and field normalizers (mean, stddev) are learned (saved in `data/column_normalizers.csv`) and used to compute z-scores that are fed into sigmoid mapping.

---

## Evaluation & tests

* Sample-based evaluation routines are provided:

  * `run_recommendation_tests_sample` — runs a sampling holdout over many users and reports hit-rates and average club precision/recall.
  * `evaluate_recommenders_holdout` — holdout-style evaluator that can be used for offline metrics.
* Default evaluation mode in `main` runs a sample test (configurable by sample size and `k`).
* Evaluators hold out a fraction of each user’s friends (1/4) and check whether held-out friends appear in top-K recommendations.

---

## Interactive terminal UI

* Simple Windows console UI (no POSIX dependencies).
* On start it prints a stylized title ("Pokec") and shows the number of loaded users.
* Prompts for a user ID to inspect; prints user summary (id, age, gender, clubs, friends).
* Menu options (numbered) let you choose:

  * graph-based friend recommendations
  * collaborative friend recommendations
  * interest-based friend recommendations
  * collaborative club recommendations
* After each recommendation the UI prints ranked ids and scores and returns to the user menu.
* UI is intentionally simple and Windows-only (uses `system("cls")` for clearing screen).

---

## Data files (where to place them)

Place the following files under the project `data/` directory:

* `data/soc-pokec-profiles.txt` — raw Pokec profiles (original dataset).
* `data/soc-pokec-relationships.txt` — raw edge list.
* `data/lem-me-sk.bin` — Slovak lemmatizer binary used by the lemmatizer wrapper (placed in `data/`).
* `config/text_columns.txt` — list of text columns to consider for TF–IDF (one per line).

The program creates / expects derived files (if present they are loaded to skip recomputation):

* `data/adjacency.csv` — serialized adjacency graph.
* `data/users_encoded.csv` — encoded user profiles (tokens -> ids).
* `data/column_normalizers.csv` — computed column normalizers (mean / std).
* `data/median_age.txt` — cached median age.

---

## Installation & build on Windows (step-by-step)

1. **Prerequisites**

   * Visual Studio 2022 or 2023 with C++ workload (MSVC toolchain).
   * CMake (>= 3.20 recommended).
   * Git.
   * Sufficient RAM & disk (dataset + derived files are large).

2. **Clone repository**

```
git clone https://github.com/pymlex/recommendation-system-pokec.git
cd recommendation-system-pokec
```

3. **Prepare data**

   * Put the Pokec dataset files into `data/`:

     * `soc-pokec-profiles.txt`
     * `soc-pokec-relationships.txt`
   * Put the lemmatizer binary into `data/lem-me-sk.bin`.
   * Ensure `config/text_columns.txt` exists (the repo may include a sample).

4. **Configure & build**

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

* If you prefer a particular generator (Visual Studio), you can run `cmake -G "Visual Studio 17 2022" ..` then open the generated solution.

5. **Run**

```
cd build/Release
kurs.exe
```

* Program will ask how many users to load (enter `100000` or `0` for all).
* Follow interactive prompts.

---

## About the modified lemmagen

This project uses a **modified C++ lemmagen** included in `third_party/lemmagen`. The included binary dictionary `data/lem-me-sk.bin` contains Slovak lemmas derived from the Python lemmagen resources and is embedded as the lemmatizer data file. The project ships an adapted native C++ lemmagen wrapper (`lemmatizer_wrapper.*`) that loads `lem-me-sk.bin`.

---

## Practical tips & troubleshooting (Windows)

* Clean build if you had earlier multiple definitions or linking errors: delete `build/` fully then re-run CMake.
* If MSVC linker complains about multiple definitions, ensure only one translation unit defines a symbol (the repo is structured to avoid ODR but old object files in `build/` may persist).
* The first run may be I/O / CPU heavy: vocab building, graph serialization and encoding can take significant time. Once `data/adjacency.csv` and `data/users_encoded.csv` are created, subsequent runs are much faster.
* If memory becomes an issue, load fewer users at startup (program prompts for number to load).

---

## Final notes

* The design prioritises speed and memory-efficiency in practice: IDF precomputation + neighborhood-limited scanning avoids an all-to-all similarity computation.
* The Fill-Aware Similarity (FAS) explicitly accounts for how much of a profile is filled to avoid overconfident recommendations based on very few shared fields.
* The repository contains evaluation harnesses and test runners so you can measure hit@k and club recommendation quality on samples.

---

Enjoy exploring Pokec recommendations.
