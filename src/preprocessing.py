import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Lowercase text and remove punctuation"""
    if pd.isna(text) or not text:
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.strip() and t not in string.punctuation and t not in STOPWORDS]
    return tokens