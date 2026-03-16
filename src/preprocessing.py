import nltk
from nltk.tokenize import word_tokenize
import string
import pandas as pd

nltk.download('punkt', quiet=True)

def preprocess_keep_capitals(text):
    if pd.isna(text) or not text:
        return []
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.strip() and t not in string.punctuation]
    return tokens