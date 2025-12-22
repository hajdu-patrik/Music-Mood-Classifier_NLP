import numpy as np
import time
import re
import os
import json
import gzip
from scipy.sparse import load_npz

# --- Configuration & Paths ---
# Determine the absolute paths dynamically to ensure it works on any server (e.g., Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

# --- Data Files ---
# These files are optimized for size (JSON + GZIP) to stay under the 250MB serverless limit
LYRICS_METADATA_PATH = os.path.join(MODEL_DIR, 'lyrics_metadata.json.gz')
LYRICS_MATRIX_PATH = os.path.join(MODEL_DIR, 'lyrics_tfidf_matrix.npz')

def get_recommendations(artist, song, lyrics_data, tfidf_matrix, top_n=5):
    """
    Finds songs with similar lyrics that match the mood of the input song.
    Uses native Matrix Multiplication instead of Scikit-learn for memory efficiency.
    """
    try:
        # 1. Search in the metadata list (Linear Search)
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
        
        # 3. Similarity calculation (optimized for Light-weight Deployment)
        # Instead of using sklearn.metrics.pairwise.cosine_similarity, use Scipy's native dot product.
        # Since TF-IDF vectors are normalized, Cosine Similarity == Dot Product.
        
        # Extract the vector for the target song (this is a Sparse Row Vector)
        song_vector = tfidf_matrix[target_index]
        
        # Calculate Dot Product: (Matrix) . (Vector Transposed)
        # Result: A column vector containing similarity scores for ALL songs.
        dot_product = tfidf_matrix.dot(song_vector.T)
        
        # Convert the sparse result to a dense numpy array and flatten it to 1D
        sim_scores = dot_product.toarray().flatten()
        
        # 4. Filtering and Sorting strategy
        candidates = []
        for i, score in enumerate(sim_scores):
            if i == target_index: continue # Skip the input song itself
            
            if lyrics_data[i]['emotion_label'] == target_emotion_label:
                candidates.append((i, score))
        
        # Sort candidates by similarity score in descending order (highest similarity first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select Top N recommendations
        top_candidates = candidates[:top_n]
        
        # 5. Result formatting for the Frontend
        recommendations = []
        for idx, score in top_candidates:
            # Copy the metadata and add the calculated similarity score
            rec_item = lyrics_data[idx].copy()
            rec_item['similarity'] = float(score) # Convert numpy.float to python float for JSON serialization
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
    """
    Loads the optimized data files into memory.
    1. Metadata (JSON.GZ) -> List of Dicts
    2. TF-IDF Matrix (NPZ) -> Scipy Sparse Matrix
    """
    print("\nLoading processed data (Lightweight Mode)...")
    start_time = time.time()
    
    try:
        # 1. Load Metadata
        # 'wt' mode in gzip means text mode, allowing direct JSON parsing
        with gzip.open(LYRICS_METADATA_PATH, "rt", encoding="utf-8") as f:
            lyrics_data = json.load(f)

        # 2. Load TF-IDF Matrix
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
    """
    Entry point for initializing the application state.
    On Vercel (read-only system), regeneration is disabled.
    """
    if force_regenerate:
        print("Warning: Regeneration is not supported in Vercel/Lightweight mode.")
    return load_artifacts()

def main_cli():
    # Helper for local testing via CLI
    artifacts = load_artifacts()
    if artifacts:
        print("System loaded. Ready.")
    else:
        print("Failed to load system.")