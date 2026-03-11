# 🎬 Movie Recommendation Clustering System

This project builds a **movie recommendation system using unsupervised machine learning**. Movies are grouped into clusters based on **genre, ratings, and popularity**.

The clustering approach helps identify similar movies so that recommendations can be generated for users.

---

## 🌐 Live Application

🔗 **Streamlit App:**
https://movierecommandation-rsiwamtnxmzdmchurkhiih.streamlit.app/

---

## 📂 GitHub Repository

🔗 **Project Repository:**
https://github.com/Bhuvana-ai-ml/movie_recommandation.git

---

## 📊 Dataset

Dataset used: **The Movies Dataset from Kaggle**

Main features used:

* Title
* Genres
* Vote Average (rating)
* Popularity

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## 🧠 Machine Learning Techniques

The project applies **unsupervised learning algorithms**.

### 1️⃣ KMeans Clustering

Groups movies into clusters based on similarity.

### 2️⃣ Hierarchical Clustering

Alternative clustering technique using a tree structure.

### 3️⃣ Feature Engineering

Genres are converted into numerical form using **MultiLabelBinarizer**.

### 4️⃣ Feature Scaling

**StandardScaler** is used to normalize the dataset.

### 5️⃣ Dimensionality Reduction

**PCA and t-SNE** are used for cluster visualization.

---

## 📈 Workflow

Dataset → Data Cleaning → Genre Encoding → Feature Scaling → Clustering → Visualization

---

## 🚀 Streamlit Application

The Streamlit app allows users to:

* Select a movie
* Identify its cluster
* Get recommendations of similar movies

---

## ▶️ How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Keerthan D**
B.Tech CSE (AI & ML)

Machine Learning Project – Movie Recommendation Clustering System
