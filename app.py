# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load Mood Dataset
mood_data_url = 'https://raw.githubusercontent.com/Akane-Asahi/mood-book/refs/heads/main/mood_classification_dataset.csv'
df_moods = pd.read_csv(mood_data_url)
label_encoder = LabelEncoder()
df_moods['encoded_mood'] = label_encoder.fit_transform(df_moods['mood'])

# Load Book Dataset
book_data_url = 'https://raw.githubusercontent.com/Akane-Asahi/mood-book/refs/heads/main/book_recommendation_dataset.csv'
df_books = pd.read_csv(book_data_url)

# Load Model & Train Classifier
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(df_moods['text'].tolist(), convert_to_tensor=False)
y = df_moods['encoded_mood']
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X, y)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_input = data.get("prompt", "")

    # Predict mood
    embedding = model.encode([user_input])
    probabilities = clf.predict_proba(embedding)[0]
    top_indices = torch.topk(torch.tensor(probabilities), 3).indices.tolist()
    top_moods = label_encoder.inverse_transform(top_indices)

    # Confirm mood (auto pick top)
    confirmed_mood = top_moods[0]

    # Recommend books
    mood_books = df_books[df_books['mood'] == confirmed_mood]
    if mood_books.empty:
        return jsonify({"mood": confirmed_mood, "books": []})

    descriptions = mood_books['description'].tolist()
    book_titles = mood_books['book_name'].tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k = torch.topk(similarity_scores, k=3)

    recommended_books = [book_titles[idx] for idx in top_k.indices]

    return jsonify({
        "mood": confirmed_mood,
        "books": recommended_books,
        "alternatives": list(top_moods)
    })

@app.route("/", methods=["GET"])
def home():
    return "Mood Book Recommender API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
