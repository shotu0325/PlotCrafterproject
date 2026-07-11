from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load data ----------
books = pd.read_csv("books.csv", low_memory=False)
books["content"] = (
    books["Book-Title"].fillna("").astype(str) + " " +
    books["Book-Author"].fillna("").astype(str) + " " +
    books["Publisher"].fillna("").astype(str)
)

# ---------- TF-IDF for recommendations ----------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["content"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

app = Flask(__name__)

# ---------- Reviews database setup ----------
DB_NAME = "reviews.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isbn TEXT NOT NULL,
            reviewer_name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review_text TEXT,
            date_posted TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_reviews(isbn):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM reviews WHERE isbn = ? ORDER BY id DESC", (isbn,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_average_rating(isbn):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT AVG(rating), COUNT(*) FROM reviews WHERE isbn = ?", (isbn,))
    avg, count = c.fetchone()
    conn.close()
    if avg is None:
        return None, 0
    return round(avg, 1), count

def attach_ratings(book_list):
    """Adds avg_rating and review_count keys to each book dict in a list."""
    for book in book_list:
        avg, count = get_average_rating(book.get("ISBN"))
        book["avg_rating"] = avg
        book["review_count"] = count
    return book_list

def get_books_by_genre(genre, n=10):
    mask = books["Genre"].fillna("").str.contains(genre, case=False, na=False)
    subset = books[mask]
    result = subset.head(n).to_dict('records')
    return attach_ratings(result)

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
        science=nonfiction
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

    result = candidates.head(5).to_dict('records')
    result = attach_ratings(result)
    return render_template("result.html", results=result)

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
    recommended_list = attach_ratings(recommended.to_dict('records'))

    selected_dict = selected_book.iloc[0].to_dict()
    avg, count = get_average_rating(isbn)
    selected_dict["avg_rating"] = avg
    selected_dict["review_count"] = count

    reviews = get_reviews(isbn)

    return render_template(
        "similar.html",
        results=recommended_list,
        selected=selected_dict,
        reviews=reviews,
        avg_rating=avg,
        review_count=count
    )

@app.route("/add_review/<isbn>", methods=['POST'])
def add_review(isbn):
    name = request.form.get('reviewer_name', '').strip()
    rating = request.form.get('rating')
    text = request.form.get('review_text', '').strip()
    next_page = request.form.get('next')

    if not name or not rating:
        return redirect(next_page or url_for('index'))

    try:
        rating = int(rating)
        rating = max(1, min(5, rating))
    except ValueError:
        return redirect(next_page or url_for('index'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reviews (isbn, reviewer_name, rating, review_text, date_posted) VALUES (?, ?, ?, ?, ?)",
        (isbn, name, rating, text, datetime.now().strftime("%d %b %Y"))
    )
    conn.commit()
    conn.close()

    if next_page:
        return redirect(next_page)
    return redirect(url_for('similar_books', isbn=isbn))

@app.route('/timer')
def timer():
    return render_template("timer.html")

if __name__ == '__main__':
    app.run(debug=True)