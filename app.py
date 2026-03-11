import streamlit as st
import pandas as pd
import numpy as np
import ast
import kagglehub

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation Clustering System")

st.write("This app clusters movies based on genre, rating and popularity.")

# -------------------------------
# DATA LOADING (CACHED)
# -------------------------------

@st.cache_data
def load_data():

    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

    df = pd.read_csv(
        f"{path}/movies_metadata.csv",
        engine="python",
        on_bad_lines="skip"
    )

    df = df[['title','genres','vote_average','popularity']]

    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

    df.dropna(inplace=True)

    return df


df = load_data()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

@st.cache_data
def preprocess_data(df):

    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return [i['name'] for i in genres]
        except:
            return []

    df['genres'] = df['genres'].apply(extract_genres)

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])

    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    features = pd.concat(
        [genre_df, df[['vote_average','popularity']]],
        axis=1
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    return df, X


df, X = preprocess_data(df)

# -------------------------------
# MODEL TRAINING (CACHED)
# -------------------------------

@st.cache_resource
def train_model(X):

    kmeans = KMeans(n_clusters=5, random_state=42)

    clusters = kmeans.fit_predict(X)

    return kmeans, clusters


kmeans, clusters = train_model(X)

df['cluster'] = clusters

# -------------------------------
# MOVIE RECOMMENDATION
# -------------------------------

st.sidebar.header("🎥 Select Movie")

movie_name = st.sidebar.selectbox(
    "Choose a movie",
    sorted(df['title'].unique())
)

if st.sidebar.button("Recommend Movies"):

    cluster = df[df['title'] == movie_name]['cluster'].values[0]

    recommendations = df[df['cluster'] == cluster]['title'].head(10)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("•", movie)

# -------------------------------
# CLUSTER VISUALIZATION
# -------------------------------

st.subheader("📊 Movie Clusters Visualization")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

fig = plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['cluster']
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Movie Clusters")

st.pyplot(fig)

# -------------------------------
# DATA PREVIEW
# -------------------------------

with st.expander("View Dataset"):

    st.dataframe(df.head(50))
