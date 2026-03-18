import pandas as pd
import json
from collections import defaultdict
import pickle
import os

from src.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer

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
inverted_index = defaultdict(set)
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

    text = f"{titles[i]} {doc}".strip() # combine title and descr to use all info for embedding and indexing

    document_corpus[doc_id] = {
        "title": titles[i],
        "description": doc,
        "id": doc_id
    }

    # Tokenize the text and update the inverted index
    tokens = preprocess_text(doc)
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].add(doc_id)

    embeddings_dict[doc_id] = model.encode(text, convert_to_numpy=True)

# Sorting and filtering the corpus
def sort_key(epid):
    season, number = epid.split('x')
    return int(season), int(number)

sorted_corpus = dict(sorted(document_corpus.items(), key=lambda x: sort_key(x[0])))

#Getting rid of season 11
filtered_corpus = {k: v for k, v in sorted_corpus.items() if not k.startswith("11x")}
filtered_doc_ids = list(filtered_corpus.keys())

filtered_embeddings = {
    doc_id: embeddings_dict[doc_id]
    for doc_id in filtered_doc_ids
}

filtered_inverted_index = {
    token: [doc_id for doc_id in doc_ids if doc_id in filtered_doc_ids]
    for token, doc_ids in inverted_index.items()
}

# Saving the corpus and inverted index in JSON format for later use 
with open(os.path.join(DW_DATA, "document_corpus_dw.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_corpus, f, ensure_ascii=False, indent=2)

with open(os.path.join(DW_DATA, "inverted_index.json"), "w", encoding="utf-8") as f:
    json.dump(dict(filtered_inverted_index), f, ensure_ascii=False, indent=2)

print(f"Corpus created: {len(filtered_corpus)} episodes saved.")


"""SQLite DB"""

import sqlite3

DB_PATH = os.path.join(DW_DATA, "doctor_who.db")

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

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

for doc_id, doc in filtered_corpus.items():
    cur.execute("INSERT OR REPLACE INTO episodes VALUES (?, ?, ?)",
                (doc_id, doc['title'], doc['description']))

for token, doc_ids in filtered_inverted_index.items():
    for doc_id in doc_ids:
        cur.execute("INSERT INTO inverted_index VALUES (?, ?)", (token, doc_id))

for doc_id, emb in filtered_embeddings.items():
    cur.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (doc_id, sqlite3.Binary(pickle.dumps(emb))))

con.commit()
con.close()

"""FAISS index creation"""

import numpy as np
import faiss

doc_ids = filtered_doc_ids
embedding_matrix = np.vstack([filtered_embeddings[doc_id] for doc_id in doc_ids]).astype("float32")

# normalize for cosine similarity
faiss.normalize_L2(embedding_matrix)

dimension = embedding_matrix.shape[1]

# === Option A: exact search
# index = faiss.IndexFlatIP(dimension)

# === Option B: ANN with HNSW
import os
os.environ["OMP_NUM_THREADS"] = "1"
index = faiss.IndexHNSWFlat(dimension, 32)  # cosine via inner product
index.hnsw.efConstruction = 200

index.add(embedding_matrix)

# Create mappings between doc_id and index position for retrieval
index_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
doc_id_to_index = {doc_id: i for i, doc_id in index_to_doc_id.items()}

# save index
faiss.write_index(index, os.path.join(DW_DATA, "faiss.index"))

# save mappings
with open(os.path.join(DW_DATA, "faiss_mapping.json"), "w") as f:
    json.dump(index_to_doc_id, f)

print("FAISS index created and saved.")