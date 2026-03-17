from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text
import sqlite3
import numpy as np

"""Return tokenized texts and corresponding doc IDs"""
def build_bm25_corpus(document_corpus):
    texts = []
    doc_ids = []
    for doc_id, doc in document_corpus.items():
        texts.append(preprocess_text(f"{doc['title']} {doc['description']}"))
        doc_ids.append(doc_id)
    return texts, doc_ids

"""Return top_n document IDs ranked by BM25"""
def bm25_search(query: str, texts, doc_ids, top_n=5):
    bm25 = BM25Okapi(texts)
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]

def build_bm25_corpus_sqlite(conn):
    """
    Gets all documents from SQLite, tokenizes them, and returns texts and doc_ids.
    """
    cur = conn.cursor()
    cur.execute("SELECT doc_id, title, description FROM episodes")
    rows = cur.fetchall()
    
    texts = []
    doc_ids = []
    for doc_id, title, description in rows:
        texts.append(preprocess_text(f"{title} {description}"))
        doc_ids.append(doc_id)
    
    return texts, doc_ids


def bm25_search_sqlite(query: str, conn, top_n=5):
    """
    Performs BM25 search based on a query using the SQLite database.
    """
    texts, doc_ids = build_bm25_corpus_sqlite(conn)
    bm25 = BM25Okapi(texts)
    
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]