# Professional Streamlit Book Recommendation System
# With Custom CSS Styling & Professional UI/UX

import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches
from datetime import datetime

# ---------------------------------
# CONFIG
# ---------------------------------
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"


# ---------------------------------
# CUSTOM CSS
# ---------------------------------
def inject_custom_css():
    custom_css = """
    <style>
    /* Main Theme */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --dark-bg: #0F1419;
        --card-bg: #1A1F2E;
        --text-primary: #FFFFFF;
        --text-secondary: #B0B8C1;
        --success-color: #06D6A0;
        --danger-color: #EF476F;
    }

    /* Hide Streamlit Branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Overall Background */
    .stApp {
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 100%);
        color: var(--text-primary);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F2E 0%, #0F1419 100%);
        border-right: 2px solid var(--primary-color);
    }

    /* Title Styling */
    h1 {
        color: var(--primary-color);
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(46, 134, 171, 0.3);
        margin-bottom: 10px;
    }

    h2 {
        color: var(--secondary-color);
        font-size: 1.8rem;
        font-weight: 700;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 10px;
        margin-top: 30px;
    }

    h3 {
        color: var(--accent-color);
        font-weight: 600;
    }

    /* Search Input */
    .stTextInput > div > div > input {
        background-color: var(--card-bg);
        color: var(--text-primary);
        border: 2px solid var(--primary-color);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 15px rgba(241, 143, 1, 0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Cards Container */
    .book-card {
        background: linear-gradient(135deg, var(--card-bg), rgba(46, 134, 171, 0.1));
        border: 2px solid rgba(46, 134, 171, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .book-card:hover {
        border-color: var(--accent-color);
        box-shadow: 0 8px 32px rgba(241, 143, 1, 0.2);
        transform: translateY(-4px);
    }

    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.2), rgba(162, 59, 114, 0.2));
        border: 2px solid var(--primary-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: var(--text-primary);
        margin: 10px 5px;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--accent-color);
    }

    .stat-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Info/Success/Error Messages */
    .stAlert {
        border-radius: 12px;
        border: 2px solid;
        backdrop-filter: blur(10px);
    }

    .stAlert > div {
        padding: 16px;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }

    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: var(--card-bg);
        color: var(--text-primary);
        border: 2px solid var(--primary-color);
        border-radius: 8px;
        padding: 10px;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 2px solid rgba(46, 134, 171, 0.3);
        margin: 20px 0;
    }

    /* Liked Books Sidebar */
    .liked-book-item {
        background: rgba(6, 214, 160, 0.1);
        border-left: 4px solid var(--success-color);
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    /* Recommendations Grid */
    .rec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: var(--accent-color);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 8px;
        margin-top: 8px;
    }

    /* Loading Animation */
    .loading-spinner {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


inject_custom_css()


# ---------------------------------
# Functions
# ---------------------------------

def fetch_books(query: str, max_results: int = 40):
    params = {"q": query, "maxResults": max_results}
    try:
        res = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data.get("items", [])
        return []
    except:
        return []


def extract_book_info(items):
    books = []
    for item in items:
        info = item.get("volumeInfo", {})
        books.append({
            "id": item.get("id"),
            "title": info.get("title", "No Title"),
            "authors": ", ".join(info.get("authors", [])),
            "description": info.get("description", ""),
            "categories": ", ".join(info.get("categories", [])),
            "thumbnail": info.get("imageLinks", {}).get("thumbnail", ""),
            "published_date": info.get("publishedDate", "N/A"),
            "page_count": info.get("pageCount", "N/A"),
            "rating": info.get("averageRating", "N/A")
        })
    return pd.DataFrame(books)


def best_title_match(title, df):
    matches = get_close_matches(title, df["title"].tolist(), n=1, cutoff=0.4)
    if matches:
        return df[df["title"] == matches[0]].index[0]
    return None


def build_tfidf(df):
    df["text"] = (
            df["title"].fillna("") + " " +
            df["authors"].fillna("") + " " +
            df["description"].fillna("") + " " +
            df["categories"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["text"])
    return matrix


def recommend(df, tfidf_matrix, book_index, top_k):
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine").fit(tfidf_matrix)
    distances, indices = nn.kneighbors(tfidf_matrix[book_index])
    return indices[0][1:]


def display_book_card(book, key_prefix=""):
    with st.container():
        st.markdown(f'<div class="book-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            if book["thumbnail"]:
                st.image(book["thumbnail"], width=140, use_container_width=False)
            else:
                st.markdown('<div style="background: #333; width: 140px; height: 200px; border-radius: 8px;"></div>',
                            unsafe_allow_html=True)

        with col2:
            st.markdown(f'<h3 style="margin-top: 0;">{book["title"]}</h3>', unsafe_allow_html=True)

            st.markdown(f'<span class="badge">📖 {book["authors"]}</span>', unsafe_allow_html=True)

            if book["rating"] != "N/A":
                st.markdown(f'<span class="badge">⭐ {book["rating"]}/5</span>', unsafe_allow_html=True)

            st.markdown(
                f'<p style="color: #B0B8C1; font-size: 0.9rem;">{book["published_date"]} • Pages: {book["page_count"]}</p>',
                unsafe_allow_html=True)

            st.markdown(f'<p style="color: #B0B8C1; line-height: 1.6;">{book["description"][:280]}...</p>',
                        unsafe_allow_html=True)

            st.markdown(f'<p><span class="badge" style="background: #06D6A0;">{book["categories"]}</span></p>',
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Book Recommendation System", layout="wide", initial_sidebar_state="expanded")

# Header with gradient
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>📚 Intelligent Book Recommendation System</h1>
    <p style="color: #B0B8C1; font-size: 1.1rem;">Discover your next favorite book with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

if "liked_books" not in st.session_state:
    st.session_state["liked_books"] = []
if "scroll_to_search" not in st.session_state:
    st.session_state["scroll_to_search"] = False
if "show_rec" not in st.session_state:
    st.session_state["show_rec"] = False
if "show_explore" not in st.session_state:
    st.session_state["show_explore"] = False

st.markdown("---")

# Search Section
st.markdown('<h2>🔍 Search Books</h2>', unsafe_allow_html=True)

search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    query = st.text_input("", placeholder="Search by title, author, or keyword (e.g., machine learning, Stephen King)",
                          label_visibility="collapsed")

with search_col2:
    search_button = st.button("🔍 Search", use_container_width=True)

if search_button and query:
    with st.spinner("🔎 Searching for books..."):
        raw = fetch_books(query)
        df = extract_book_info(raw)

        if df.empty:
            st.error("❌ No results found. Try a different search term.")
        else:
            st.session_state["df"] = df
            st.success(f"✅ Found {len(df)} books matching '{query}'")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.markdown('<h2>📘 Search Results</h2>', unsafe_allow_html=True)

    for idx, (_, row) in enumerate(df.iterrows()):
        display_book_card(row, key_prefix=f"search_{idx}")

        lk = f"like_{row['id']}"
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("❤️ Like", key=lk, use_container_width=True):
                if row.to_dict() not in st.session_state["liked_books"]:
                    st.session_state["liked_books"].append(row.to_dict())
                    st.success("✅ Added to liked books!")
                    st.rerun()
                else:
                    st.info("📌 Already in liked books")

        st.divider()

    # Recommendation Section
    if st.session_state["liked_books"]:
        st.markdown('<h2>⭐ Smart Recommendations</h2>', unsafe_allow_html=True)

        liked_df = pd.DataFrame(st.session_state["liked_books"])
        titles = liked_df["title"].tolist()

        rec_col1, rec_col2 = st.columns([2, 1])

        with rec_col1:
            selected_title = st.selectbox("Select a liked book for recommendations", titles,
                                          label_visibility="collapsed")

        with rec_col2:
            top_k = st.slider("# of Recommendations", 3, 20, 6, label_visibility="collapsed")

        if st.button("🎯 Generate Recommendations", use_container_width=True):
            with st.spinner("🤖 Generating AI recommendations..."):
                # Build combined dataframe
                combined_books = list(df.to_dict(orient='records')) + st.session_state["liked_books"]
                combined_df = pd.DataFrame(combined_books).drop_duplicates(subset=['id'], keep='first')

                if len(combined_df) < 2:
                    st.warning("Need more books to generate recommendations.")
                else:
                    tfidf_matrix = build_tfidf(combined_df)
                    idx = best_title_match(selected_title, combined_df)

                    if idx is not None:
                        rec_idxs = recommend(combined_df, tfidf_matrix, idx, top_k)

                        st.markdown(f'<h3>📚 Recommended for you based on "{selected_title}"</h3>',
                                    unsafe_allow_html=True)

                        for i, rec_idx in enumerate(rec_idxs, 1):
                            st.markdown(f'<p style="color: var(--accent-color); font-weight: bold;">#{i}</p>',
                                        unsafe_allow_html=True)
                            book = combined_df.iloc[rec_idx]
                            display_book_card(book, key_prefix=f"rec_{i}")

                    else:
                        st.error("Could not match this book. Try another selection.")

# Browse by Categories Section
st.markdown('<h2>📚 Browse by Categories</h2>', unsafe_allow_html=True)

categories = {
    "Fiction": "fiction",
    "Science Fiction": "science fiction",
    "Mystery": "mystery thriller",
    "Romance": "romance",
    "Business": "business",
    "Self-Help": "self-help",
    "History": "history",
    "Biography": "biography",
}

cat_col1, cat_col2 = st.columns([3, 1])

with cat_col1:
    selected_category = st.selectbox("Choose a category to explore", list(categories.keys()), key="cat_select",
                                     label_visibility="collapsed")

with cat_col2:
    if st.button("Browse 📖", use_container_width=True):
        with st.spinner(f"Loading {selected_category} books..."):
            raw = fetch_books(categories[selected_category], max_results=10)
            cat_df = extract_book_info(raw)
            st.session_state["cat_df"] = cat_df

if "cat_df" in st.session_state and not st.session_state["cat_df"].empty:
    st.markdown(f'<h3>Books in {selected_category}</h3>', unsafe_allow_html=True)

    for idx, (_, row) in enumerate(st.session_state["cat_df"].iterrows()):
        display_book_card(row, key_prefix=f"cat_{idx}")

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("❤️ Like", key=f"like_cat_{row['id']}", use_container_width=True):
                if row.to_dict() not in st.session_state["liked_books"]:
                    st.session_state["liked_books"].append(row.to_dict())
                    st.success("✅ Added to liked books!")
                    st.rerun()
                else:
                    st.info("📌 Already in liked books")

        st.divider()

# Sidebar - Features & Liked Books with Hover Effects
st.sidebar.markdown("""
<h2 style="color: var(--primary-color); border-bottom: 3px solid var(--primary-color); padding-bottom: 10px;">✨ Features</h2>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
.feature-card {
    background: rgba(46, 134, 171, 0.1);
    border: 2px solid var(--primary-color);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.feature-card:hover {
    background: rgba(46, 134, 171, 0.25);
    border-color: var(--accent-color);
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
}

.feature-card-2 {
    background: rgba(162, 59, 114, 0.1);
    border: 2px solid var(--secondary-color);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.feature-card-2:hover {
    background: rgba(162, 59, 114, 0.25);
    border-color: var(--accent-color);
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(162, 59, 114, 0.3);
}

.feature-card-3 {
    background: rgba(241, 143, 1, 0.1);
    border: 2px solid var(--accent-color);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.feature-card-3:hover {
    background: rgba(241, 143, 1, 0.25);
    border-color: var(--success-color);
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(241, 143, 1, 0.3);
}

.feature-card-4 {
    background: rgba(6, 214, 160, 0.1);
    border: 2px solid var(--success-color);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.feature-card-4:hover {
    background: rgba(6, 214, 160, 0.25);
    border-color: var(--primary-color);
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(6, 214, 160, 0.3);
}
</style>

<div class="feature-card">
    <p style="color: var(--accent-color); font-weight: bold; margin: 0;">🔍 Active Search</p>
    <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 5px 0 0 0;">Google Books API</p>
</div>

<div class="feature-card-2">
    <p style="color: var(--accent-color); font-weight: bold; margin: 0;">⭐ Recommendations</p>
    <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 5px 0 0 0;">TF-IDF & ML Engine</p>
</div>

<div class="feature-card-3">
    <p style="color: var(--accent-color); font-weight: bold; margin: 0;">📖 Browse Categories</p>
    <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 5px 0 0 0;">Explore by Genres</p>
</div>

<div class="feature-card-4">
    <p style="color: var(--success-color); font-weight: bold; margin: 0;">❤️ Liked Books</p>
    <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 5px 0 0 0;">Your Collection</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<h2 style="color: var(--success-color); border-bottom: 3px solid var(--success-color); padding-bottom: 10px;">❤️ Your Liked Books</h2>
""", unsafe_allow_html=True)

if st.session_state["liked_books"]:
    st.sidebar.markdown(
        f'<p style="color: var(--accent-color); font-weight: bold;">Total: {len(st.session_state["liked_books"])} books</p>',
        unsafe_allow_html=True)

    for idx, lb in enumerate(st.session_state["liked_books"]):
        st.sidebar.markdown(f'''
        <div class="liked-book-item">
            <strong>{lb['title']}</strong>
            <br><small>{lb['authors']}</small>
        </div>
        ''', unsafe_allow_html=True)

        if st.sidebar.button(f"Remove", key=f"remove_{idx}", use_container_width=True):
            st.session_state["liked_books"].pop(idx)
            st.rerun()
else:
    st.sidebar.info("❌ No liked books yet. Like books to build your collection!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<p style="text-align: center; color: var(--text-secondary); font-size: 0.85rem;">
    🚀 <strong>Powered by AI</strong><br>
    Built with TF-IDF & Machine Learning<br>
    <br>
    <strong>Features:</strong><br>
    🔍 Google Search<br>
    ⭐ Smart Recommendations<br>
    📖 Category Explorer<br>
    ❤️ Liked Books Collection
</p>
""", unsafe_allow_html=True)