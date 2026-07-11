from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


books = pd.read_csv("books.csv", low_memory=False)
books["content"] = (
    books["Book-Title"].fillna("").astype(str) + " " +
    books["Book-Author"].fillna("").astype(str) + " " +
    books["Publisher"].fillna("").astype(str)
)


vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["content"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

app = Flask(__name__)

def get_books_by_genre(genre, n=10):
    
    mask = books["Genre"].fillna("").str.contains(genre, case=False, na=False)
    subset = books[mask]
    return subset.head(n).to_dict('records')

@app.route('/')
def index():
    fantasy = get_books_by_genre("Science Fiction & Fantasy")
    mystery = get_books_by_genre("Mystery & Thriller")
    romance = get_books_by_genre("Romance")
    nonfiction = get_books_by_genre("Non-Fiction")

    return render_template(
        "index.html",
        fantasy=fantasy,
        mystery=mystery,
        romance=romance,
        nonfiction=nonfiction   
    )

@app.route("/genre/<genre>")
def genre(genre):
    result = get_books_by_genre(genre, n=20)
    return {"books": result}

@app.route('/recommend', methods=['POST'])
def recommend():
    book = request.form['book_name']
    genre_filter = request.form.get('genre') 

    query_vector = vectorizer.transform([book])
    if query_vector.nnz == 0:
        return render_template("result.html", results=[], error="No matching books found. Try a different title.")

    similarity = cosine_similarity(query_vector, tfidf_matrix)
    indices = similarity.argsort()[0][::-1]

    candidates = books.iloc[indices]
    if genre_filter:
        candidates = candidates[candidates["Genre"].fillna("").str.contains(genre_filter, case=False, na=False)]

    result = candidates.head(5)
    return render_template("result.html", results=result.to_dict('records'))

@app.route("/similar/<isbn>")
def similar_books(isbn):
    selected_book = books[books["ISBN"] == isbn]
    if selected_book.empty:
        return "Book not found", 404

    book_index = selected_book.index[0]
    book_vector = tfidf_matrix[book_index]
    similarity = cosine_similarity(book_vector, tfidf_matrix)
    indices = similarity.argsort()[0][::-1]
    top_indices = indices[1:4]

    recommended = books.iloc[top_indices].copy()
    scores = similarity[0][top_indices]
    recommended["Similarity Score"] = scores * 100

    return render_template(
        "similar.html",
        results=recommended.to_dict('records'),
        selected=selected_book.iloc[0]
    )

@app.route('/timer')
def timer():
    return render_template("timer.html")

if __name__ == '__main__':
    app.run(debug=True)