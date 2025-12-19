import pandas as pd
import numpy as np
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

import os
import joblib
import argparse
from scipy.sparse import save_npz, load_npz

nltk_data_path = "/tmp"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    stop_words = set(stopwords.words('english')) 
except Exception as e:
    print(f"WARNING: NLTK download failed: {e}. Fallback to basic processing.")
    stop_words = set()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

LYRICS_DATA_PATH = os.path.join(PROJECT_ROOT, 'sources', 'spotify_millsongdata.csv')
EMOTIONS_DATA_PATH = os.path.join(PROJECT_ROOT, 'sources', 'emotions.csv')

EMOTION_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger',  4: 'fear', 5: 'surprise'
}

MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
PIPELINE_CACHE_DIR = os.path.join(MODEL_DIR, 'pipeline_cache')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_pipeline.joblib')
LYRICS_DF_PATH = os.path.join(MODEL_DIR, 'analyzed_lyrics.pkl')
LYRICS_MATRIX_PATH = os.path.join(MODEL_DIR, 'lyrics_tfidf_matrix.npz')
LYRICS_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'lyrics_vectorizer.joblib')


def load_data():
    print("Loading raw datasets...")
    try:
        emotions_df = pd.read_csv(EMOTIONS_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: '{EMOTIONS_DATA_PATH}' not found.")
        return None, None

    try:
        lyrics_df = pd.read_csv(LYRICS_DATA_PATH)
        lyrics_df = lyrics_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    except FileNotFoundError:
        print(f"ERROR: '{LYRICS_DATA_PATH}' not found.")
        return None, None

    print(f"Loaded: {len(emotions_df)} emotion samples.")
    print(f"Loaded: {len(lyrics_df)} lyrics samples).")
    
    lyrics_df.dropna(subset=['text', 'artist', 'song'], inplace=True)
    emotions_df.dropna(subset=['text', 'label'], inplace=True)
    
    return lyrics_df, emotions_df


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    punct = set(string.punctuation)
    
    clean_tokens = [
        token for token in tokens 
        if token not in stop_words and token not in punct and token.isalpha()
    ]
    
    return " ".join(clean_tokens)


def train_emotion_model(emotions_df):
    print("\nTraining emotion classification model...")
    X = emotions_df['text']
    y = emotions_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    try:
        os.makedirs(PIPELINE_CACHE_DIR, exist_ok=True)
        memory = PIPELINE_CACHE_DIR
    except OSError:
        memory = None

    emotion_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ], memory=memory)
    
    start_time = time.time()
    emotion_pipeline.fit(X_train, y_train)
    print(f"Model training complete. Duration: {time.time() - start_time:.2f} sec.")
    
    report = "Training complete."
    if len(X_test) > 0:
        y_pred = emotion_pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=[EMOTION_MAP[i] for i in range(6)])
    
    return emotion_pipeline, report


def analyze_lyrics_database(lyrics_df, emotion_model):
    print("\nAnalyzing lyrics database...")
    lyrics_df['emotion_label'] = emotion_model.predict(lyrics_df['text'])
    lyrics_df['emotion_name'] = lyrics_df['emotion_label'].map(EMOTION_MAP)
    return lyrics_df


def create_similarity_matrix(lyrics_df):
    print("\nBuilding similarity matrix...")
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    lyrics_tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics_df['text'])
    return lyrics_tfidf_matrix, tfidf_vectorizer


def get_recommendations(artist, song, lyrics_df, tfidf_matrix, top_n=5):
    try:
        song_data = lyrics_df[
            (lyrics_df['artist'].str.lower() == artist.lower()) &
            (lyrics_df['song'].str.lower() == song.lower())
        ]
        
        if song_data.empty:
            return {"error": f"Sorry, the song '{artist} - {song}' was not found in the database."}

        song_entry = song_data.iloc[0]
        song_index = song_entry.name
        target_emotion_label = song_entry['emotion_label']
        target_emotion_name = song_entry['emotion_name']
        
        song_vector = tfidf_matrix[song_index]
        sim_scores = cosine_similarity(song_vector, tfidf_matrix)

        indices = np.where(lyrics_df['emotion_label'] == target_emotion_label)[0]
        
        relevant_scores = sim_scores[0][indices]
        
        best_indices_local = np.argsort(relevant_scores)[::-1]
        best_indices_global = indices[best_indices_local]
        
        best_indices_global = best_indices_global[best_indices_global != song_index][:top_n]
        
        recommendations = []
        for idx in best_indices_global:
            row = lyrics_df.iloc[idx]
            score = sim_scores[0][idx]
            recommendations.append({
                'artist': row['artist'],
                'song': row['song'],
                'similarity': score
            })

        return {
            "error": None,
            "input_song": {
                "artist": song_entry['artist'],
                "song": song_entry['song'],
                "emotion": target_emotion_name
            },
            "recommendations": recommendations
        }
                
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return {"error": f"An internal error occurred: {e}"}


def load_artifacts():
    print("\nLoading pre-trained models...")
    try:
        emotion_model = joblib.load(EMOTION_MODEL_PATH)
        lyrics_df = pd.read_pickle(LYRICS_DF_PATH)
        tfidf_matrix = load_npz(LYRICS_MATRIX_PATH)
        tfidf_vectorizer = joblib.load(LYRICS_VECTORIZER_PATH)
        
        return {
            "emotion_model": emotion_model,
            "lyrics_df": lyrics_df,
            "tfidf_matrix": tfidf_matrix,
            "tfidf_vectorizer": tfidf_vectorizer
        }
    except Exception as e:
        print(f"Error loading model files: {e}.")
        return None

def regenerate_artifacts():
    print("WARNING: Artifacts not found! Attempting regeneration...")
    
    if not os.access(MODEL_DIR, os.W_OK):
        print("ERROR: Read-only file system detected. Cannot retrain model on server.")
        return None

    print("--- Starting Model Regeneration (Local Environment) ---")
    lyrics_df_raw, emotions_df = load_data()
    if lyrics_df_raw is None: return None
    
    emotion_model, _ = train_emotion_model(emotions_df)
    
    joblib.dump(emotion_model, EMOTION_MODEL_PATH)
    lyrics_df = analyze_lyrics_database(lyrics_df_raw, emotion_model)
    lyrics_df.to_pickle(LYRICS_DF_PATH)
    tfidf_matrix, tfidf_vectorizer = create_similarity_matrix(lyrics_df)
    save_npz(LYRICS_MATRIX_PATH, tfidf_matrix)
    joblib.dump(tfidf_vectorizer, LYRICS_VECTORIZER_PATH)

    return {
        "emotion_model": emotion_model,
        "lyrics_df": lyrics_df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer
    }


def initialize_app(force_regenerate=False):
    model_files = [EMOTION_MODEL_PATH, LYRICS_DF_PATH, LYRICS_MATRIX_PATH, LYRICS_VECTORIZER_PATH]
    all_files_exist = all(os.path.exists(f) for f in model_files)

    artifacts = None
    if all_files_exist and not force_regenerate:
        artifacts = load_artifacts()

    if artifacts is None:
        print("Artifacts missing or reload forced. Regenerating...")
        artifacts = regenerate_artifacts()
    
    return artifacts