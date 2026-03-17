import pandas as pd
import json
from collections import defaultdict
import pickle
import os
from src.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer, util

# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DW_DATA = os.path.join(PROJECT_ROOT, "dw_data")

# Load CSV files
df_details = pd.read_csv(os.path.join(DW_DATA, "all-detailsepisodes.csv"))
df_imdb = pd.read_csv(os.path.join(DW_DATA, "imdb_details.csv"))

# Merge datasets on title
df = pd.merge(df_details, df_imdb, on="title", how="inner")
df.to_csv(os.path.join(DW_DATA, "merged_dataset.csv"), index=False)

# Initialize the inverted index and document corpus
inverted_index = defaultdict(list)
embeddings_dict = {}
document_corpus = {}

model = SentenceTransformer('all-MiniLM-L6-v2')

# Preparing the corpus for building the inverted index
descriptions = df['description'].fillna("").tolist()
episode_keys = (df['season'].astype(str) + "x" + df['number'].astype(str)).tolist()
titles = df['title'].tolist()

# Building corpus and inverted index
for i, doc in enumerate(descriptions):
    doc_id = episode_keys[i] # e.g. "1x1"
    document_corpus[doc_id] = {
        "title": titles[i],
        "description": doc,
        "id": doc_id
    }
    tokens = preprocess_text(doc)
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

    embeddings_dict[doc_id] = model.encode(f"{titles[i]} {doc}", convert_to_numpy=True)

# Sorting and filtering the corpus
def sort_key(epid):
    season, number = epid.split('x')
    return int(season), int(number)

sorted_corpus = dict(sorted(document_corpus.items(), key=lambda x: sort_key(x[0])))

#Getting rid of season 11
filtered_corpus = {k: v for k, v in sorted_corpus.items() if not k.startswith("11x")}

# Saving the corpus and inverted index
with open(os.path.join(DW_DATA, "document_corpus_dw.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_corpus, f, ensure_ascii=False, indent=2)

with open(os.path.join(DW_DATA, "inverted_index.json"), "w", encoding="utf-8") as f:
    json.dump(dict(inverted_index), f, ensure_ascii=False, indent=2)

print(f"Corpus created: {len(filtered_corpus)} episodes saved, {len(document_corpus) - len(filtered_corpus)} removed.")






"""SQLite database creation"""

import sqlite3

DB_PATH = os.path.join(DW_DATA, "doctor_who.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# episodes
cur.execute("""
CREATE TABLE IF NOT EXISTS episodes (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT
)
""")

#inverted_index

cur.execute("""
CREATE TABLE IF NOT EXISTS inverted_index (
    token TEXT,
    doc_id TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    doc_id TEXT PRIMARY KEY,
    embedding BLOB
)
""")

for doc_id, doc in document_corpus.items():
    cur.execute("INSERT OR REPLACE INTO episodes VALUES (?, ?, ?)",
                (doc_id, doc['title'], doc['description']))

for token, doc_ids in inverted_index.items():
    for doc_id in doc_ids:
        cur.execute("INSERT INTO inverted_index VALUES (?, ?)", (token, doc_id))

for doc_id, emb in embeddings_dict.items():
    cur.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (doc_id, sqlite3.Binary(pickle.dumps(emb))))

conn.commit()