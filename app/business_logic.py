# --- Importing to read input files, create vectors, and train models ---
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

# --- Importing for saving and loading Python objects and handling command line arguments  ---
import os
import joblib
import argparse
from scipy.sparse import save_npz, load_npz


# To start the program:
# 1. Step: Create a .venv virtual enviroment with this command: "/c/Users/XXX/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv"
# 2. Step: Run .venv (python's virtual enviroment) with this command: "source .venv/Scripts/activate"
# 3. Step: Download the packages that the program uses with this command: "pip install -r imports/requirements.txt"
# 4. Step: Run main.py file with this command: "python app/app.py"
# 5. Step (Optional): To force re-training, run: "python app/app.py --regenerate"


# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Paths relative to the parent folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Dataset paths
LYRICS_DATA_PATH = os.path.join(PROJECT_ROOT, 'sources', 'spotify_millsongdata.csv')
EMOTIONS_DATA_PATH = os.path.join(PROJECT_ROOT, 'sources', 'emotions.csv')

# Emotion label mapping
EMOTION_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger',  4: 'fear', 5: 'surprise'
}

# Saved model and data access paths
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
PIPELINE_CACHE_DIR = os.path.join(MODEL_DIR, 'pipeline_cache')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_pipeline.joblib')
LYRICS_DF_PATH = os.path.join(MODEL_DIR, 'analyzed_lyrics.pkl')
LYRICS_MATRIX_PATH = os.path.join(MODEL_DIR, 'lyrics_tfidf_matrix.npz')
LYRICS_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'lyrics_vectorizer.joblib')


def load_data():
    print("Loading raw datasets...")
    
    # Load emotions
    try:
        emotions_df = pd.read_csv(EMOTIONS_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: '{EMOTIONS_DATA_PATH}' not found.")
        return None, None

    # Load lyrics
    try:
        lyrics_df = pd.read_csv(LYRICS_DATA_PATH)
        # Using 1.0 frac to load all data, as in the original script.
        lyrics_df = lyrics_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    except FileNotFoundError:
        print(f"ERROR: '{LYRICS_DATA_PATH}' not found.")
        return None, None

    print(f"Loaded: {len(emotions_df)} emotion samples.")
    print(f"Loaded: {len(lyrics_df)} lyrics samples).")
    
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
    print("This might take a few minutes...")
    
    X = emotions_df['text']
    y = emotions_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs(PIPELINE_CACHE_DIR, exist_ok=True)
    emotion_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ], memory=PIPELINE_CACHE_DIR)
    
    start_time = time.time()
    emotion_pipeline.fit(X_train, y_train)
    print(f"Model training complete. Duration: {time.time() - start_time:.2f} sec.")
    
    # --- Model Evaluation ---
    # We only run evaluation if the test set is not empty
    if len(X_test) > 0:
        y_pred = emotion_pipeline.predict(X_test)
        print("\n--- Model Evaluation ---")
        report = classification_report(y_test, y_pred, target_names=[EMOTION_MAP[i] for i in range(6)])
        print(report)
        print("-----------------------------------------------------")
    else:
        print("\n--- Model Evaluation Skipped ---")
        report = "No evaluation performed."
    
    return emotion_pipeline, report


def analyze_lyrics_database(lyrics_df, emotion_model):
    print("\nAnalyzing lyrics database with the emotion model...")
    print("This might take a few minutes...")

    start_time = time.time()
    
    lyrics_df['emotion_label'] = emotion_model.predict(lyrics_df['text'])
    lyrics_df['emotion_name'] = lyrics_df['emotion_label'].map(EMOTION_MAP)
    
    print(f"Lyrics analysis complete. Duration: {time.time() - start_time:.2f} sec.")
    
    print("\nEmotion distribution in the lyrics database (Top 5):")
    print(lyrics_df['emotion_name'].value_counts().head())
    
    return lyrics_df


def create_similarity_matrix(lyrics_df):
    print("\nBuilding similarity matrix (TF-IDF) from lyrics...")
    
    # We use preprocess_text here to clean up the lyrics as well.
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    
    lyrics_tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics_df['text'])
    
    print(f"Similarity matrix ready. Shape: {lyrics_tfidf_matrix.shape}")
    
    return lyrics_tfidf_matrix, tfidf_vectorizer


def get_recommendations(artist, song, lyrics_df, tfidf_matrix, top_n=5):
    try:
        # 1. Find the song (case-insensitive)
        song_data = lyrics_df[
            (lyrics_df['artist'].str.lower() == artist.lower()) &
            (lyrics_df['song'].str.lower() == song.lower())
        ]
        
        if song_data.empty:
            # Return an error dictionary
            return {"error": f"Sorry, the song '{artist} - {song}' was not found in the database."}

        # 2. Extract song data
        song_entry = song_data.iloc[0]
        song_index = song_entry.name
        target_emotion_label = song_entry['emotion_label']
        target_emotion_name = song_entry['emotion_name']
        
        # 3. Calculate similarity
        song_vector = tfidf_matrix[song_index]
        sim_scores = cosine_similarity(song_vector, tfidf_matrix)
        lyrics_df['similarity'] = sim_scores[0]
        
        # 4. Filter and sort recommendations
        recommendations_df = lyrics_df[
            (lyrics_df['emotion_label'] == target_emotion_label) &
            (lyrics_df.index != song_index)
        ].sort_values(by='similarity', ascending=False)
        
        # 5. Prepare the results
        top_recs = recommendations_df.head(top_n)[['artist', 'song', 'similarity']]
        
        # Clean up the temporary column
        lyrics_df.drop(columns=['similarity'], inplace=True, errors='ignore')

        # Return a success dictionary
        return {
            "error": None,
            "input_song": {
                "artist": song_entry['artist'],
                "song": song_entry['song'],
                "emotion": target_emotion_name
            },
            "recommendations": top_recs.to_dict('records') # Convert DataFrame to list of dicts
        }
                
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        # Return an error dictionary
        return {"error": f"An internal error occurred: {e}"}


def load_artifacts():
    print("\nLoading pre-trained models and processed data...")
    start_time = time.time()
    
    try:
        emotion_model = joblib.load(EMOTION_MODEL_PATH)
        lyrics_df = pd.read_pickle(LYRICS_DF_PATH)
        tfidf_matrix = load_npz(LYRICS_MATRIX_PATH)
        tfidf_vectorizer = joblib.load(LYRICS_VECTORIZER_PATH)
        
        print(f"Models loaded successfully. Duration: {time.time() - start_time:.2f} sec.")
        
        # Return all loaded artifacts in a dictionary for easy passing
        return {
            "emotion_model": emotion_model,
            "lyrics_df": lyrics_df,
            "tfidf_matrix": tfidf_matrix,
            "tfidf_vectorizer": tfidf_vectorizer
        }
    except Exception as e:
        print(f"Error loading model files: {e}. Forcing regeneration.")
        return None

def regenerate_artifacts():
    print("--- Starting Model Regeneration ---")

    # 2.1 Loading raw data
    lyrics_df_raw, emotions_df = load_data()
    if lyrics_df_raw is None or emotions_df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # 2.2 Teaching and saving emotion models
    emotion_model, _ = train_emotion_model(emotions_df)

    joblib.dump(emotion_model, EMOTION_MODEL_PATH)
    print(f"Emotion model saved to {EMOTION_MODEL_PATH}")

    # 2.3 Analyzing and saving the lyrics database
    lyrics_df = analyze_lyrics_database(lyrics_df_raw, emotion_model)
    lyrics_df.to_pickle(LYRICS_DF_PATH)
    print(f"Analyzed lyrics database saved to {LYRICS_DF_PATH}")

    # 2.4 Building and saving a similarity matrix
    tfidf_matrix, tfidf_vectorizer = create_similarity_matrix(lyrics_df)
    save_npz(LYRICS_MATRIX_PATH, tfidf_matrix)
    joblib.dump(tfidf_vectorizer, LYRICS_VECTORIZER_PATH)
    print(f"Similarity matrix saved to {LYRICS_MATRIX_PATH}")
    print(f"Lyrics TF-IDF vectorizer saved to {LYRICS_VECTORIZER_PATH}")
    
    print("--- Model Regeneration Complete ---")
    
    # Return all generated artifacts
    return {
        "emotion_model": emotion_model,
        "lyrics_df": lyrics_df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer
    }


def initialize_app(force_regenerate=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_files = [
        EMOTION_MODEL_PATH, LYRICS_DF_PATH,
        LYRICS_MATRIX_PATH, LYRICS_VECTORIZER_PATH
    ]
    all_files_exist = all(os.path.exists(f) for f in model_files)

    artifacts = None
    if all_files_exist and not force_regenerate:
        artifacts = load_artifacts()

    if artifacts is None: # Ha a betöltés hibás volt, vagy kérték a regenerálást
        artifacts = regenerate_artifacts()
    
    if artifacts is None:
        # Ha a regenerálás sem sikerült (pl. nincsenek CSV-k)
        raise RuntimeError("Could not load or regenerate artifacts. Check source files.")
        
    return artifacts


def run_interactive_loop(artifacts):
    print("\n--- Music Mood Classifier System Started ---")
    print("To exit, type: 'exit'")
    
    # Unpack the artifacts
    lyrics_df = artifacts["lyrics_df"]
    tfidf_matrix = artifacts["tfidf_matrix"]
    
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
            tfidf_matrix
        )

def main_cli():
    # 1. Reading command line arguments
    parser = argparse.ArgumentParser(description="Music Mood Classifier System")
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="Force regeneration of all models and processed data."
    )
    args = parser.parse_args()

    # Ensure the 'model' directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Loading or generating models
    model_files = [
        EMOTION_MODEL_PATH,
        LYRICS_DF_PATH,
        LYRICS_MATRIX_PATH,
        LYRICS_VECTORIZER_PATH
    ]
    all_files_exist = all(os.path.exists(f) for f in model_files)

    artifacts = None
    if all_files_exist and not args.regenerate:
        artifacts = load_artifacts()

    if artifacts is None: # This triggers if loading failed or regeneration was forced
        artifacts = regenerate_artifacts()

    # 3. Run interactive loop if artifacts are ready
    if artifacts:
        run_interactive_loop(artifacts)
    else:
        print("Could not load or regenerate artifacts. Exiting.")

if __name__ == "__main__":
    main_cli()