

import re
import joblib
import pandas as pd
import numpy as np

print("Loading trained model...")
genre_model = joblib.load("genre_classifier.pkl")
genre_vectorizer = joblib.load("tfidf_vectorizer_genre.pkl")
genre_names = joblib.load("genre_names.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("Loading books.csv...")
books = pd.read_csv("books.csv", low_memory=False)
print("Columns found:", list(books.columns))
print("Total books:", len(books))


books["content"] = (
    books["Book-Title"].fillna("").astype(str) + " " +
    books["Book-Author"].fillna("").astype(str) + " " +
    books["Publisher"].fillna("").astype(str)
)
books["clean_text"] = books["content"].apply(clean_text)
print("Vectorizing text...")
X = genre_vectorizer.transform(books["clean_text"])

print("Predicting probabilities for all books (this is the slow-ish part, but should take well under a minute)...")
probs = genre_model.predict_proba(X)  

THRESHOLD = 0.3

print("Converting probabilities into genre labels...")
genre_lists = []
for row in probs:
    predicted = [genre_names[i] for i, p in enumerate(row) if p > THRESHOLD]
    if not predicted:
        predicted = [genre_names[np.argmax(row)]] 
    genre_lists.append(", ".join(predicted))

books["Genre"] = genre_lists


books.drop(columns=["clean_text", "content"], inplace=True)
books.to_csv("books.csv", index=False)

print("\nDone! Genre column added and saved to books.csv")
print(books[["Book-Title", "Genre"]].head(10).to_string())