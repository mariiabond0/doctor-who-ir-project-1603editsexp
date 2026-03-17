import os
import json
from src.boolean_search import boolean_search, boolean_search_sqlite
from src.bm_25 import build_bm25_corpus, build_bm25_corpus_sqlite, bm25_search, bm25_search_sqlite
from src.sentence_transformers import encode_corpus, load_embeddings_from_db, semantic_search, semantic_search_sqlite
import pandas as pd
import sqlite3

# Prepare database connection
DB_PATH = "dw_data/doctor_who.db"
conn = sqlite3.connect(DB_PATH)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DW_DATA = os.path.join(PROJECT_ROOT, "dw_data")
CORPUS_PATH = os.path.join(DW_DATA, "document_corpus_dw.json")
INDEX_PATH = os.path.join(DW_DATA, "inverted_index.json")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    document_corpus = json.load(f)

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

# Queries and expected answers
queries = [
    "Doctor fights with Weeping Angels and wants to save Amy",
    "Doctor and Clara in nineteenth century",
    "Doctor meets River Song for the first time",
    "Doctor and Donna encounter Daleks and Davros",
    "Doctor and Rose end up in a parallel universe",
    "Doctor meets van Gogh",
    "Doctor and Martha face time paradox creatures or similar threats",
    "Doctor and Bill encounter Cybermen",
    "Paternoster Gang and Doctor and Vastra and Strax",
    "Doctor and Rose meets her father Pete Tyler"
]

answers = [
    ["5x4", "5x5", "7x5", "6x11", "3x10"],
    ["7x6", "7x12", "7x8", "1x3", "7x10"],
    ["4x8", "4x9", "5x4", "5x5", "6x1"],
    ["4x12", "4x13", "2x12", "2x13", "9x1"],
    ["2x5", "2x6", "2x12", "2x13", "4x11"],
    ["5x10", "5x1", "5x12", "5x13", "1x3"],
    ["3x10", "3x8", "3x9", "3x11", "3x12"],
    ["10x11", "10x12", "2x5", "2x6", "8x12"],
    ["7x6", "7x11", "7x13", "8x1", "6x7"],
    ["1x8", "2x5", "2x6", "1x13", "4x11"]
]

# Prepare BM25 and Semantic embeddings
texts, doc_ids = build_bm25_corpus_sqlite(conn)
corpus_embeddings = load_embeddings_from_db(conn)

def evaluate_method(method_name, query_func):
    print(f"\n--- {method_name} ---")
    for i, query in enumerate(queries):
        results = query_func(query)
        print(f"Query {i+1}: {len(set(results) & set(answers[i]))}/{len(answers[i])}")

# Boolean search wrapper
def boolean_query(q):
    return boolean_search_sqlite(q, conn)

# BM25 search wrapper
def bm25_query(q):
    return bm25_search_sqlite(q, conn)

# Semantic search wrapper
def semantic_query(q):
    return semantic_search_sqlite(q, conn)

# Run evaluation
evaluate_method("Boolean Search", boolean_query)
evaluate_method("BM25 Search", bm25_query)
evaluate_method("Semantic Search", semantic_query)

results_dict = {
    "Boolean Correct": [],
    "BM25 Correct": [],
    "ST Correct": []
}

for i, query in enumerate(queries):
    results_dict["Boolean Correct"].append(len(set(boolean_query(query)) & set(answers[i])))
    results_dict["BM25 Correct"].append(len(set(bm25_query(query)) & set(answers[i])))
    results_dict["ST Correct"].append(len(set(semantic_query(query)) & set(answers[i])))

df = pd.read_csv("dw_data/search_results_summary.csv")
for col, vals in results_dict.items():
    df[col] = vals

df.to_csv("dw_data/search_results_summary.csv", index=False)
