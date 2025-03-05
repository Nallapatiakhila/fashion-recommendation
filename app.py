from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
df = pd.read_csv("myntra_products_catalog.csv")
df['Description'].fillna("", inplace=True)

# Train recommendation model
tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(df['Description'])
nn_model = NearestNeighbors(n_neighbors=6, metric='cosine').fit(features)

def recommend_outfits(query, top_n=5):
    query_vec = tfidf.transform([query])
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_n+1)
    return df.iloc[indices[0][1:]][['ProductName', 'Description', 'PrimaryColor']]

@app.route('/')
def home():
    return "Fashion Recommendation System is Running! Use /recommend endpoint."

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    results = recommend_outfits(query)
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
