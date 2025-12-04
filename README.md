# Pokec Recommendation System

<img width="345" height="126" alt="pokec" src="https://github.com/user-attachments/assets/952bd94b-4ddd-4d02-bb43-d8960b75cb9c" />

A **C++ heuristic recommendation system** built for the Slovak social network **Pokec** (≈1.6M users). The core is implemented in modern C++ and remains responsible for all heavy work: graph processing, TF–IDF text similarity, and the Fill-Aware Similarity metric used to recommend friends and clubs. A **FastAPI** wrapper (Python) runs the C++ executable as a subprocess, exposes HTTP endpoints that return JSON, and serves a small static HTML UI. Optionally the server can be made publicly reachable via **ngrok** (token provided in a config).

## Key features

* **Friend recommendation** from partial user registration (friends + friends-of-friends).
* **Collaborative friend** recommendation using friend → friend-of-friend propagation weighted by similarities.
* **Interest-based** friend recommendation (profile similarity using TF–IDF for text columns + other structured fields).
* **Collaborative club** (subscription) recommendations.
* **Fill-Aware Similarity** ($FAS$) — similarity measure that accounts both for per-field similarity and how many fields are actually filled in (profile completion awareness).

## Fill-Aware Similarity ($FAS$) Metrics

For two profiles define per-field similarity $s_i$ for all fields present in both profiles; aggregate them into $S$ (average of bounded per-field scores) and compute fill factor $F$ as the ratio of common non-empty fields to the total possible fields. The final metric $FAS$ combines $S$ and $F$ harmonically:

$$FAS = \frac{2 \cdot S \cdot F}{S + F}$$

This allows the introduction of a penalty for similarity with sparsely filled profiles and addresses the problem of sparsity.

## Field Similarities

* Per-field similarities: TF–IDF cosine for text columns, normalized set overlap for clubs/friends, region match for hierarchical fields, ratio for numeric fields (age/completion).
* Each raw $s_i$ is transformed via a sigmoid of a z-score computed using column/field normalizers (if available) before averaging into $S$.
* IDF for text columns is precomputed once and stored in the recommender for efficient scoring.

## Project structure

```
.
├── .kaggle
├── build
├── config
├── data
│   └── explore
│   ├── ...
├── include
│   ├── bin_reader.h
│   ├── recommender.h
│   ├── ...
│   ├── tfidf_index.h
│   └── vocab_builder.h
├── src
│   ├── bin_reader.cpp
│   ├── evaluator.cpp
│   ├── ...
│   ├── user_profile.cpp
│   └── vocab_builder.cpp
├── python    
│   ├── templates/         
│   ├── app.py
│   └── config.yaml
└── third_party
    ├── lemmagen  (modified lemmagen C++ with slovak dict)
    └── matplotplusplus
```

## Data used:

Before the first run, use the `download_pokec.h` to download the raw Pokec datasets files:

* `data/soc-pokec-profiles.txt` (raw Pokec profiles).
* `data/soc-pokec-relationships.txt` (raw edge list).
* `config/text_columns.txt` (one text column name per line).

As most of the fields are text, we use the TF-IDF and bag-of-words approach for text processing. Lemmatization is based on the [lemmagen-c](https://github.com/evillique/lemmagen-c) project with a Slovak [vocabulary ](https://pypi.org/project/Lemmagen/), download the latter manually. The file structure for lemmatization will be:

* `third_party/lemmagen` contains a modified native C++ lemmatizer used with `data/lem-me-sk.bin` (Slovak dictionary).

## Architecture & runtime flow

1. **C++ backend (`api_cli.exe`)** — loads encoded users, adjacency and normalizers into memory; implements recommendation algorithms. The C++ process accepts textual commands on stdin (for example `USER {id}`) and writes JSON responses to stdout. This keeps all core logic in C++ unchanged.

2. **Python FastAPI wrapper** — launches and monitors the C++ process, exposes HTTP endpoints, parses C++ JSON responses, and serves the static HTML UI. Optionally opens an ngrok tunnel for external access.

## Execution sequence

1. **Read config** — load list of text columns from `config/text_columns.txt`.
2. **Tokenizer & lemmatizer** initialisation (`Tokenizer`, `Lemmatiser` using `data/lem-me-sk.bin`).
3. **Vocabulary** (`VocabBuilder`) load or build. If missing, a pass over raw profiles builds token vocabularies.
4. **Graph** (`GraphBuilder`) load or build, then convert to adjacency list `adj_list`.
5. **Encode users** — produce `data/users_encoded.csv` if missing.
6. **Load users** — `load_users_encoded(...)` reads `users_encoded.csv`. The program (or wrapper) offers a `load_users` config parameter (e.g. `100000` or `0` = all).
7. **Data cleanup** — compute/load `median_age` and fill missing ages.
8. **Column normalizers** — load or compute `data/column_normalizers.csv`.
9. **Recommender init** — instantiate `Recommender`, set normalizers, compute IDF per text column and set internal text column list.

## API endpoints

* `GET /api/user/{uid}` — full user object plus recommendations (graph, collaborative, interest, clubs). Response is JSON matching the terminal UI output.
* `GET /api/recommend/graph/{uid}?topk=...`
* `GET /api/recommend/collab/{uid}?topk=...`
* `GET /api/recommend/interest/{uid}?topk=...`
* `GET /api/recommend/clubs/{uid}?topk=...`
* `GET /health` — basic health check and configured `load_users`.

## Build & run (Windows)

```
git clone https://github.com/pymlex/recommendation-system-pokec.git
cd recommendation-system-pokec
```

Load the data, then build the C++ part: 

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Set Python `venv` and install the requirements:

```
cd ..\python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Set you `ngrok` authtoken in `python/config.yaml`. Run the backend:

```
python python/app.py
```

## Interactive terminal UI (kept for local inspection)

The C++ project also ships a simple Windows console UI. The FastAPI wrapper is the primary runtime mode; the terminal UI remains available if you run the native executable directly.
