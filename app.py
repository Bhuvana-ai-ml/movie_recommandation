import streamlit as st
import pandas as pd
import numpy as np
import ast

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation Clustering System")

st.write(
"""
This application groups movies into clusters based on
**genres, rating, and popularity**, then recommends similar movies.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("movies_small.csv")

    df = df[['title','genres','vote_average','popularity']]

    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

    df.dropna(inplace=True)

    return df


df = load_data()

# ---------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------

@st.cache_data
def preprocess(df):

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


df, X = preprocess(df)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

@st.cache_resource
def train_model(X):

    model = KMeans(n_clusters=5, random_state=42)

    clusters = model.fit_predict(X)

    return model, clusters


model, clusters = train_model(X)

df['cluster'] = clusters

# ---------------------------------------------------
# SIDEBAR MOVIE SELECTION
# ---------------------------------------------------

st.sidebar.header("🎥 Select a Movie")

movie = st.sidebar.selectbox(
    "Choose a movie",
    sorted(df['title'].unique())
)

# ---------------------------------------------------
# MOVIE RECOMMENDATION
# ---------------------------------------------------

if st.sidebar.button("Recommend Movies"):

    cluster = df[df['title'] == movie]['cluster'].values[0]

    recommendations = df[df['cluster'] == cluster]['title'].head(10)

    st.subheader("🎬 Recommended Movies")

    for m in recommendations:
        st.write("•", m)

# ---------------------------------------------------
# CLUSTER VISUALIZATION
# ---------------------------------------------------

st.subheader("📊 Movie Cluster Visualization")

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

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

with st.expander("View Dataset"):

    st.dataframe(df.head(50))
