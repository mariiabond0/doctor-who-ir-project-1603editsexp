# Doctor Who Information Retrieval System

This project implements an information retrieval system based on the **Doctor Who** TV series. It supports multiple search methods, including **Boolean search**, **BM25**, and **semantic search with Sentence Transformers**, allowing users to query episodes based on text descriptions, titles, and metadata.

## Features

* **Corpus Creation**

  * Collects episode data from CSV files (`all-detailsepisodes.csv`, `imdb_details.csv`)
  * Preprocesses text (tokenization, optional stopword removal, preserving proper nouns)
  * Builds a **document corpus** and an **inverted index**

* **Search Methods**

  * **Boolean Search** using an inverted index
  * **BM25 Search** using `rank_bm25`
  * **Semantic Search** using `SentenceTransformers` (`all-MiniLM-L6-v2`)

* **Storage Options**

  * JSON files (`document_corpus_dw.json`, `inverted_index.json`)
  * SQLite database (`doctor_who.db`) containing episodes, inverted index, and precomputed embeddings
  * Optional support for pickled embeddings

* **Evaluation**

  * Supports test queries with expected answers
  * Computes overlap and correctness metrics for all search methods

## Project Structure

```
doctor-who-ir-project/
│
├─ src/
│   ├─ preprocessing.py       # Text preprocessing functions
│   ├─ creating_corpus.py     # Build corpus, inverted index, embeddings, SQLite DB
│   ├─ boolean_search.py      # Boolean search functions (JSON & SQLite)
│   ├─ bm_25.py               # BM25 functions (JSON & SQLite)
│   ├─ sentence_transformers.py # Semantic search & embedding functions
│
├─ dw_data/
│   ├─ all-detailsepisodes.csv
│   ├─ imdb_details.csv
│   ├─ document_corpus_dw.json
│   ├─ inverted_index.json
│   ├─ doctor_who.db
│   └─ search_results_summary.csv
│
├─ main.py                   # Run evaluation and query searches
└─ README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/doctor-who-ir-project.git
cd doctor-who-ir-project
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

**Dependencies include:**

* `pandas`
* `nltk`
* `rank_bm25`
* `sentence-transformers`
* `numpy`

3. Download NLTK punkt tokenizer:

```python
import nltk
nltk.download('punkt')
```

## Usage

1. **Build corpus and database**:

```bash
python src/creating_corpus.py
```

2. **Run evaluation of search methods**:

```bash
python main.py
```

3. **Query examples**:

```python
from src.boolean_search import boolean_search_sqlite
from src.bm_25 import bm25_search_sqlite
from src.sentence_transformers import semantic_search, load_embeddings_from_db
import sqlite3

conn = sqlite3.connect("dw_data/doctor_who.db")
embeddings = load_embeddings_from_db(conn)

query = "Doctor and Clara in nineteenth century"
results_boolean = boolean_search_sqlite(query, conn)
results_bm25 = bm25_search_sqlite(query, conn)
results_semantic = semantic_search(query, embeddings, top_n=5)
```

## Notes

* Preprocessing preserves capital letters in proper names for better retrieval of characters and places.
* BM25 and semantic search can be adapted to use either JSON files or the SQLite database.
* The SQLite database allows fast retrieval and easier management of metadata.

## License

This project is released under the MIT License.

---

If you want, I can also **add a “Quick Start” section with ready-to-run examples for all three search methods using SQLite** so anyone can test queries immediately.

Do you want me to add that?
