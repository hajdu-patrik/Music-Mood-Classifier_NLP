# Music Mood Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)

![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-2E8B57?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=flat&logo=numpy&logoColor=white)

![HTML5](https://img.shields.io/badge/Frontend-HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/Style-CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/Script-JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

![Status](https://img.shields.io/badge/Status-Educational_Project-brightgreen?style=flat)

A Python-based NLP system and web application that analyzes the emotional profile of song lyrics and recommends songs with a similar mood. The project was developed as part of the **BME-VIK "Natural Language and Semantic Technologies"** course.

The system consists of two main components:

1. **Business Logic (`business_logic.py`)** – NLP preprocessing, model training, emotion recognition, similarity calculation, and model saving.
2. **Web App (`app.py`)** – Flask-based server that serves the search interface and displays the results.

---

## 🚀 Features

- 🎭 **Emotion Classification** – 6 categories: *sadness, joy, love, anger, fear, surprise*
- 🎶 **Similarity-Based Recommendation** – based on TF-IDF + Cosine Similarity
- 🌐 **Web Interface** – clean, responsive UI
- 🌗 **Dark/Light Mode** – preference saved in browser
- ⚡ **Model Persistence** – one-time training, fast loading
- 🔁 **Force Regeneration** – retraining with `--regenerate` flag


---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+
- **Web Server:** Flask
- **ML/NLP:** Scikit-learn, TF-IDF, Logistic Regression, Cosine Similarity
- **Data:** Pandas, NumPy
- **Preprocessing:** NLTK
- **Model Saving:** Joblib, Scipy
- **Frontend:** HTML5, CSS3, JavaScript

---

## 📂 Project Structure
```
Music-Mood-Classifier/
├── app/
│ ├── business_logic.py # Core NLP/ML logic
│ └── app.py # Flask server
│
├── sources/
│ ├── spotify_millsongdata.csv
│ └── emotions.csv
│
├── imports/
│ └── requirements.txt
│
├── model/
│ └── (Generated artifacts)
│
├── static/
│ ├── style.css
│ └── scripts.js
│ └── favicon.ico
│
└── templates/
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
git clone [https://github.com/hajdu-patrik/NLP_Music-Mood-Classifier_BME_HF]
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
pip install -r imports/requirements.txt
```

### 4. Download Datasets

1.  Download the [Spotify Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) and place the `spotify_millsongdata.csv` in the root folder.
2.  Download the [Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) and place the `emotions.csv` (you may need to rename it from `text.csv`) in the root folder.

### 5. Run the Application (Two-Step Process)

#### 🔧 Step 5.1 — First-Time Setup (Training)
The --regenerate flag is mandatory when starting the model for the first time:
- load the raw CSVs
- train the emotion model
- analyze all 57k+ songs
- build the similarity matrix
- save everything to the model/ directory

```bash
python app/app.py --regenerate
```

Once the console prints this message, stop the server with Ctrl + C
```arduino
Initialization complete. Server is running.
```

#### ▶️ Step 5.2 — Run the Web App (Normal Use)
After that, you can start it without the flag:
```bash
python app/app.py
```
The server will be available here:
- http://127.0.0.1:5000
- http://localhost:5000

Open it in your browser to use it.

---

## 🎮 Console Interaction Example

```bash
--- Music Mood Recommender System Started ---
To exit, type: 'exit'

Enter artist name: abba
Enter song title: cassandra

--- Analysis: abba - cassandra ---
Determined emotion: sadness

Recommendations (based on similar emotion and text):
  1. Conway Twitty - Don't Tell Me You're Sorry (Similarity: 0.28)
  2. The Temptations - Sorry Is A Sorry Word (Similarity: 0.26)
  3. Gordon Lightfoot - Remember Me (Similarity: 0.18)
  4. Hanson - Being Me (Similarity: 0.17)
  5. Religious Music - Angels Among Us (Similarity: 0.15)

Enter artist name: exit
```

## 📊 Model Performance

The emotion classification model (Logistic Regression on TF-IDF features) achieved a **90% weighted average F1-score** on the validation set (83,362 samples).
