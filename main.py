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


# To start the program:
# 1. step: Run .venv (python's virtual enviroment) with this command: "source .venv/Scripts/activate ""
# 2. step: Run main.py file with this command: "python main.py"


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# --- Global Constants ---
# 1. Dataset paths
LYRICS_DATA_PATH = 'source/spotify_millsongdata.csv'
EMOTIONS_DATA_PATH = 'source/emotions.csv' 

# 2. Emotion label mapping
EMOTION_MAP = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def load_data():
    print("Loading datasets...")
    
    # Load emotions
    try:
        emotions_df = pd.read_csv(EMOTIONS_DATA_PATH)
        # Column names in the emotions dataset are often 'id', 'text', 'label'
        
    except FileNotFoundError:
        print(f"ERROR: '{EMOTIONS_DATA_PATH}' not found.")
        return None, None

    # Load lyrics
    try:
        lyrics_df = pd.read_csv(LYRICS_DATA_PATH)
        # Because the original dataset has 57k+ rows, 
        # you can run the program with the total amount of data specified as a percentage in the frac % variable.
        lyrics_df = lyrics_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    except FileNotFoundError:
        print(f"ERROR: '{LYRICS_DATA_PATH}' not found.")
        return None, None

    print(f"Loaded: {len(emotions_df)} emotion samples.")
    print(f"Loaded: {len(lyrics_df)} lyrics (after sampling).")
    
    # Remove missing data
    lyrics_df.dropna(subset=['text', 'artist', 'song'], inplace=True)
    emotions_df.dropna(subset=['text', 'label'], inplace=True)
    
    return lyrics_df, emotions_df


def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Removing punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    
    clean_tokens = [
        token for token in tokens 
        if token not in stop_words and token not in punct and token.isalpha()
    ]
    
    # 4. Re-join into a string for the Vectorizer
    return " ".join(clean_tokens)


def train_emotion_model(emotions_df):
    print("\nTraining emotion classification model...")
    
    X = emotions_df['text']
    y = emotions_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    emotion_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    
    start_time = time.time()
    emotion_pipeline.fit(X_train, y_train)
    print(f"Model training complete. Duration: {time.time() - start_time:.2f} sec.")
    
    y_pred = emotion_pipeline.predict(X_test)
    print("\n--- Model Evaluation (For the Report) ---")
    report = classification_report(y_test, y_pred, target_names=[EMOTION_MAP[i] for i in range(6)])
    print(report)
    print("-----------------------------------------------------")
    
    return emotion_pipeline, report


def analyze_lyrics_database(lyrics_df, emotion_model):
    print("\nAnalyzing lyrics database with the emotion model...")
    print("This might take a few minutes depending on the sample size...")
    
    start_time = time.time()
    
    lyrics_df['emotion_label'] = emotion_model.predict(lyrics_df['text'])
    lyrics_df['emotion_name'] = lyrics_df['emotion_label'].map(EMOTION_MAP)
    
    print(f"Lyrics analysis complete. Duration: {time.time() - start_time:.2f} sec.")
    
    print("\nEmotion distribution in the lyrics database (Top 5):")
    print(lyrics_df['emotion_name'].value_counts().head())
    
    return lyrics_df


def create_similarity_matrix(lyrics_df):
    print("\nBuilding similarity matrix (TF-IDF) from lyrics...")
    
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    
    lyrics_tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics_df['text'])
    
    print(f"Similarity matrix ready. Shape: {lyrics_tfidf_matrix.shape}")
    
    return lyrics_tfidf_matrix, tfidf_vectorizer


def get_recommendations(artist, song, lyrics_df, tfidf_matrix, tfidf_vectorizer, top_n=5):
    try:
        # 1. Find the song (case-insensitive)
        song_data = lyrics_df[
            (lyrics_df['artist'].str.lower() == artist.lower()) &
            (lyrics_df['song'].str.lower() == song.lower())
        ]
        
        if song_data.empty:
            print(f"\nSorry, the song '{artist} - {song}' was not found in the database.")
            print("Tip: Try to enter the name exactly as it appears in the Kaggle dataset.")
            return

        # 2. Extract song data
        song_entry = song_data.iloc[0]
        song_index = song_entry.name
        target_emotion_label = song_entry['emotion_label']
        target_emotion_name = song_entry['emotion_name']
        
        print(f"\n--- Analysis: {artist} - {song} ---")
        print(f"Determined emotion: {target_emotion_name}")
        
        # 3. Calculate similarity
        song_vector = tfidf_matrix[song_index]
        
        sim_scores = cosine_similarity(song_vector, tfidf_matrix)
        
        lyrics_df['similarity'] = sim_scores[0]
        
        # 4. Filter and sort recommendations
        recommendations = lyrics_df[lyrics_df['emotion_label'] == target_emotion_label]
        
        recommendations = recommendations[recommendations.index != song_index]
        
        recommendations = recommendations.sort_values(by='similarity', ascending=False)
        
        # 5. Display results
        print(f"\nRecommendations (based on similar emotion and text):")
        if recommendations.empty:
            print("No other similar songs found in this emotion category.")
        else:
            top_recs = recommendations.head(top_n)[['artist', 'song', 'similarity']]
            count = 1
            for _, row in top_recs.iterrows():
                print(f"  {count}. {row['artist']} - {row['song']} (Similarity: {row['similarity']:.2f})")
                count += 1
                
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")


def main_cli():
    # 1. Load data
    lyrics_df, emotions_df = load_data()
    if lyrics_df is None or emotions_df is None:
        print("Failed to load data. Exiting.")
        return
        
    # 2. Train emotion model
    emotion_model, model_report = train_emotion_model(emotions_df)
    
    # 3. Analyze lyrics database
    lyrics_df = analyze_lyrics_database(lyrics_df, emotion_model)
    
    # 4. Build similarity matrix
    tfidf_matrix, tfidf_vectorizer = create_similarity_matrix(lyrics_df)
    
    print("\n--- Music Mood Classifier System Started ---")
    print("To exit, type: 'exit'")
    
    while True:
        artist_input = input("\nEnter artist name: ").strip()
        if artist_input.lower() == 'exit':
            break
            
        song_input = input("Enter song title: ").strip()
        if song_input.lower() == 'exit':
            break

        if not artist_input or not song_input:
            print("Artist and song title cannot be empty. Please try again.")
            continue
            
        get_recommendations(
            artist_input, 
            song_input, 
            lyrics_df, 
            tfidf_matrix, 
            tfidf_vectorizer
        )

if __name__ == "__main__":
    main_cli()