import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings_from_db(conn):
    """
    Returns a dictionary {doc_id: embedding} from the embeddings table.
    """
    cur = conn.cursor()
    cur.execute("SELECT doc_id, embedding FROM embeddings")
    rows = cur.fetchall()
    embeddings_dict = {row[0]: pickle.loads(row[1]) for row in rows}
    return embeddings_dict

def semantic_search_sqlite(query: str, conn, top_n=5):
    """
    Semantic search in SQLite: fetch embeddings, calculate cosine similarity,
    and return top_n doc_ids.
    """
    embeddings_dict = load_embeddings_from_db(conn)
    doc_ids = list(embeddings_dict.keys())
    corpus_embeddings = np.stack(list(embeddings_dict.values()))  # N x D

    query_embedding = model.encode(query, convert_to_numpy=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    cosine_scores = cosine_scores.cpu().numpy() if hasattr(cosine_scores, 'cpu') else cosine_scores

    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    results = [doc_ids[i] for i in top_indices]
    return results

def encode_corpus(document_corpus):
    texts = [f"{doc['title']} {doc['description']}" for doc in document_corpus.values()]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def semantic_search(query: str, document_corpus, corpus_embeddings, top_n=5):
    query_embedding = model.encode(query, convert_to_numpy=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    cosine_scores = cosine_scores.cpu().numpy()  # ← ключевая строка
    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    doc_ids = list(document_corpus.keys())
    results = [doc_ids[i] for i in top_indices]
    return results

#def semantic_search(query: str, document_corpus, corpus_embeddings, top_n=5):
    query_embedding = model.encode(query, convert_to_numpy=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_n)[0]
    doc_ids = list(document_corpus.keys())
    results = [doc_ids[hit['corpus_id']] for hit in hits]
    return results