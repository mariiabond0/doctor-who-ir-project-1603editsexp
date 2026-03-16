import pandas as pd
from nltk.tokenize import word_tokenize
import json
from collections import defaultdict
import os
from src.preprocessing import preprocess_keep_capitals

# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DW_DATA = os.path.join(PROJECT_ROOT, "dw_data")

# Load CSV files
df_details = pd.read_csv(os.path.join(DW_DATA, "all-detailsepisodes.csv"))
df_imdb = pd.read_csv(os.path.join(DW_DATA, "imdb_details.csv"))

# Merge datasets on title
df = pd.merge(df_details, df_imdb, on="title", how="inner")
df.to_csv(os.path.join(DW_DATA, "merged_dataset.csv"), index=False)

inverted_index = defaultdict(list)
document_corpus = {}

# Preparing the corpus for building the inverted index
descriptions = df['description'].fillna("").tolist()
episode_keys = (df['season'].astype(str) + "x" + df['number'].astype(str)).tolist()
titles = df['title'].tolist()

# Filling the index
for i, doc in enumerate(descriptions):
    doc_id = episode_keys[i] # e.g. "1x1"
    document_corpus[doc_id] = {
        "title": titles[i],
        "description": doc,
        "id": doc_id
    }
    tokens = preprocess_keep_capitals(doc)
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

# Sorting and filtering the corpus
def sort_key(epid):
    season, number = epid.split('x')
    return int(season), int(number)

sorted_corpus = dict(sorted(document_corpus.items(), key=lambda x: sort_key(x[0])))

#Getting rid of season 11
filtered_corpus = {k: v for k, v in sorted_corpus.items() if not k.startswith("11x")}

with open(os.path.join(DW_DATA, "document_corpus_dw.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_corpus, f, ensure_ascii=False, indent=2)

with open(os.path.join(DW_DATA, "inverted_index.json"), "w", encoding="utf-8") as f:
    json.dump(dict(inverted_index), f, ensure_ascii=False, indent=2)

print(f"Corpus created: {len(filtered_corpus)} episodes saved, {len(document_corpus) - len(filtered_corpus)} removed.")