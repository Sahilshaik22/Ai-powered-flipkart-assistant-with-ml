import re, numpy as np, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NltkPreprocessor():
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_data(self, text: str) -> str:
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.squeeze()
        processed = []
        for txt in X:
            txt = self.clean_data(txt)
            if not txt:
                processed.append("")
                continue
            tokens = [t for t in word_tokenize(txt) if t not in self.stop_words]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            processed.append(" ".join(tokens))
        return processed