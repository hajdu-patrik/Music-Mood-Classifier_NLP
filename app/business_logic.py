import numpy as np
import time
import re
import os
import json
import gzip
from scipy.sparse import load_npz

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

# Data files
LYRICS_METADATA_PATH = os.path.join(MODEL_DIR, 'lyrics_metadata.json.gz')
LYRICS_MATRIX_PATH = os.path.join(MODEL_DIR, 'lyrics_tfidf_matrix.npz')

def get_recommendations(artist, song, lyrics_data, tfidf_matrix, top_n=5):
    try:
        # 1. Search in the metadata list
        target_index = -1
        song_entry = None
        
        artist_lower = artist.lower().strip()
        song_lower = song.lower().strip()

        for idx, item in enumerate(lyrics_data):
            if item['artist'].lower() == artist_lower and item['song'].lower() == song_lower:
                target_index = idx
                song_entry = item
                break
        
        if target_index == -1:
            return {"error": f"Sorry, the song '{artist} - {song}' was not found in the database."}

        # 2. Data extraction
        target_emotion_label = song_entry['emotion_label']
        target_emotion_name = song_entry['emotion_name']
        
        # 3. Similarity calculation (without Scikit-learn)
        # Mathematics: Cosine Similarity = (A . B) / (|A|*|B|)
        # Since our TF-IDF matrix is already normalized (L2 norm), simple multiplication (Dot Product) is sufficient.
        # Extract the vector of the target song (this is a sparse vector)
        song_vector = tfidf_matrix[target_index]
        
        # Matrix multiplication: The entire matrix multiplied by the transpose of the song vector
        # Result: A column vector containing similarity scores for all songs
        # The .dot() is a built-in fast function of the scipy sparse matrix
        dot_product = tfidf_matrix.dot(song_vector.T)
        
        # Conversion to dense array and flattening to 1D
        sim_scores = dot_product.toarray().flatten()
        
        # 4. Filtering and sorting
        candidates = []
        for i, score in enumerate(sim_scores):
            if i == target_index: continue # Skip itself
            
            if lyrics_data[i]['emotion_label'] == target_emotion_label:
                candidates.append((i, score))
        
        # Sorting by score (descending order)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Top N selection
        top_candidates = candidates[:top_n]
        
        # Result formatting
        recommendations = []
        for idx, score in top_candidates:
            rec_item = lyrics_data[idx].copy()
            rec_item['similarity'] = float(score) # float conversion for JSON compatibility
            recommendations.append(rec_item)

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
    print("\nLoading processed data (Lightweight Mode)...")
    start_time = time.time()
    
    try:
        # 1. Metadata loading from JSON
        with gzip.open(LYRICS_METADATA_PATH, "rt", encoding="utf-8") as f:
            lyrics_data = json.load(f)

        # 2. Matrix loading (Scipy sparse)
        tfidf_matrix = load_npz(LYRICS_MATRIX_PATH)
        
        print(f"Artifacts loaded successfully. Duration: {time.time() - start_time:.2f} sec.")
        
        return {
            "lyrics_df": lyrics_data,
            "tfidf_matrix": tfidf_matrix
        }
    except Exception as e:
        print(f"Error loading files: {e}.")
        return None

def initialize_app(force_regenerate=False):
    if force_regenerate:
        print("Warning: Regeneration is not supported in Vercel/Lightweight mode.")
    return load_artifacts()

def main_cli():
    artifacts = load_artifacts()
    if artifacts:
        print("System loaded. Ready.")
    else:
        print("Failed to load system.")

if __name__ == "__main__":
    main_cli()