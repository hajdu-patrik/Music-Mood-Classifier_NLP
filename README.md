![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/Math-SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![HTML5](https://img.shields.io/badge/Frontend-HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/Style-CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/Script-JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Vercel](https://img.shields.io/badge/Deployment-Vercel-000000?style=flat&logo=vercel&logoColor=white)

# Music Mood Classifier

A Python-based NLP system and web application that analyzes the emotional profile of song lyrics and recommends songs with a similar mood. The project was developed as part of the **BME-VIK "Natural Language and Semantic Technologies"** course.

> **🚀 Optimization Update:** This project has been heavily optimized for Serverless Deployment (Vercel). It runs **without** heavy libraries like Pandas, Scikit-learn, or NLTK in production, keeping the total unzipped size under 120MB.

---

## 🚀 Live Demo

**Check out the live version of the application here:**
👉 **[https://music-mood-classifier-nine.vercel.app](https://music-mood-classifier-nine.vercel.app)**

---

## 📚 Features

- 🎭 **Emotion Classification** – 6 categories: *sadness, joy, love, anger, fear, surprise*
- 🎶 **Similarity-Based Recommendation** – Uses native Matrix Multiplication (Dot Product) on TF-IDF vectors.
- 🚀 **High Performance** – Optimized for speed and low memory usage using Sparse Matrices (`.npz`) and Compressed JSON (`.json.gz`).
- 🌐 **Web Interface** – Clean, responsive UI with Dark/Light mode support.
- ⚡ **Serverless Ready** – Designed to run within strict size limits of cloud functions.

---

## 🛠️ Tech Stack

### Production (Runtime)
These libraries are used to run the web application:
- **Web Server:** Flask
- **Math & Logic:** NumPy, SciPy (Sparse Matrices)
- **Data Handling:** Native Python JSON, Gzip
- **Frontend:** HTML5, CSS3, JavaScript

### Development (Training Phase)
The models were originally trained using:
- **ML/NLP:** Scikit-learn, TF-IDF, Logistic Regression
- **Data:** Pandas, NLTK
*(Note: These are not required to run the deployed app)*

---

## 📂 Project Structure
```
Music-Mood-Classifier/
├── app/
│ ├── business_logic.py
│ ├── convert_model.py
│ └── app.py
│
├── source/
│ ├── spotify_millsongdata.csv
│ └── emotions.csv
│
├── requirements.txt
│
├── model/
│ └── (Generated artifacts)
│
├── static/
│ ├── style.css
│ └── scripts.js
│ └── favicon.ico
│
└── template/
├── index.html
└── results.html
```

---

## 💾 Datasets Used

1.  **Emotion Classification Model:** [Emotions in Text Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) (approx. 417k labeled texts).
2.  **Song Database:** [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) (approx. 57k songs with lyrics).

---

## ⚙️ Setup & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/hajdu-patrik/Music-Mood-Classifier_PYthon]
cd your-repo-name
```

### 2. Create and Activate Virtual Environment

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate"
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

Since the models are pre-trained and optimized, you can start the server immediately:
```bash
python app/app.py
```

The server will be available here:
- http://127.0.0.1:5000
- http://localhost:5000

---

## 🎮 Console Interaction Example

```bash
--- Music Mood Classifier System Started ---
To exit, type: 'exit'

Enter artist name: abba
Enter song title: cassandra

Loading processed data (Lightweight Mode)...
Artifacts loaded successfully.

Input Song: Abba - Cassandra (Emotion: sadness)
Recommendations:
 1. Conway Twitty - Don't Tell Me You're Sorry (Similarity: 0.28)
 2. The Temptations - Sorry Is A Sorry Word (Similarity: 0.26)
 ...
```

## 📊 Model Performance

The emotion classification model (Logistic Regression on TF-IDF features) achieved a **90% weighted average F1-score** on the validation set (83,362 samples).

---

## 📦 Deployment

This project is configured for automated deployment via **Vercel**.
Any push to the `main` branch automatically triggers a new build and deployment.

| Environment | Status |
| :--- | :--- |
| **Production** | [![Vercel App](https://img.shields.io/badge/Visit-Live_App-success?style=for-the-badge&logo=vercel)](https://music-mood-classifier-nine.vercel.app) |
