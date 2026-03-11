import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("🎬 Movie Recommendation Clustering")

# Load dataset
df = pd.read_csv("movies_metadata.csv", engine="python", on_bad_lines="skip")

df = df[['title','genres','vote_average','popularity']]
df.dropna(inplace=True)

# Convert popularity
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df.dropna(inplace=True)

# Extract genres
def extract_genres(x):
    try:
        genres = ast.literal_eval(x)
        return [i['name'] for i in genres]
    except:
        return []

df['genres'] = df['genres'].apply(extract_genres)

# Encode genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genres'])

genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Combine features
features = pd.concat([genre_df, df[['vote_average','popularity']]], axis=1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# UI: select movie
movie_name = st.selectbox("Select a movie", df['title'].values)

if st.button("Recommend Movies"):

    cluster = df[df['title'] == movie_name]['cluster'].values[0]

    recommendations = df[df['cluster'] == cluster]['title'].head(10)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(movie)
