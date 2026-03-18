

def faiss_search(query, top_k=10):
    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc_id = index_to_doc_id[idx]
        results.append((doc_id, float(score)))

    return results