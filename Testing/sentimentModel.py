# Testing/sentimentModel.py
import os
import pathlib
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


# --- downloads only for training/setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Testing/sentimentModel.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # ...\ai_ecom_agent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))   # make ...\ai_ecom_agent\src importable

from nlp_preprocessor import NltkPreprocessor  # <— note: no "src."


ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root (AI_ECOM_AGENT)
DATA = ROOT / "Data"
MODELS = ROOT / "models"

df = pd.read_csv(DATA / "sentiment.csv")
X = df["text"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

load = NltkPreprocessor()
X_train = load.transform(X_train)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

MODELS.mkdir(exist_ok=True)
with open(r"D:\Genai_Projects\ai_ecom_agent\models\sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
print("🎯 Saved:", MODELS / "sentiment_pipeline.pkl")
