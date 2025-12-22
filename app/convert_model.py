import pandas as pd
import json
import gzip
import os

# --- Paths ---
input_path = "model/analyzed_lyrics.pkl.gz"
output_path = "model/lyrics_metadata.json.gz"

# --- Data Loading ---
print(f"Loading: {input_path}...")

# Attempt to read the compressed pickle file
try:
    df = pd.read_pickle(input_path)
except (ValueError, OSError):
    print("Compressed file not found or invalid, trying uncompressed .pkl...")
    df = pd.read_pickle("model/analyzed_lyrics.pkl")

print(f"Original size (rows): {len(df)}")

# --- Data Conversion ---
data_list = []
for index, row in df.iterrows():
    data_list.append({
        "artist": row['artist'],
        "song": row['song'],
        "emotion_label": int(row['emotion_label']),
        "emotion_name": row['emotion_name']
    })

print(f"Saving to: {output_path}...")
# Save as GZIP compressed JSON for maximum space efficiency
with gzip.open(output_path, "wt", encoding="utf-8") as f:
    json.dump(data_list, f)

print("Done! You can now delete the old .pkl and .pkl.gz files from the model folder.")