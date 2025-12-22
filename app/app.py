from flask import Flask, render_template, request
import business_logic as bl
import argparse
import os

# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'template')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

# Initialize Flask
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# --- Global Artifacts Initialization ---
print("Flask app initializing...")
artifacts = None
try:
    artifacts = bl.initialize_app()
    print("Initialization complete.")
except Exception as e:
    print(f"CRITICAL ERROR during initialization: {e}")

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    if artifacts is None:
        return render_template('results.html', error="System is starting up or failed to load data. Please try again later.")

    artist_name = request.args.get('artist')
    song_name = request.args.get('song')

    results = None
    if artist_name and song_name:
        results = bl.get_recommendations(
            artist_name,
            song_name, 
            artifacts["lyrics_df"], 
            artifacts["tfidf_matrix"]
        )
    
    return render_template(
        'results.html', 
        artist=artist_name,
        song=song_name,
        results=results
    )

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# --- Application Startpoint (Only for local dev) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()

    if artifacts is None: 
        artifacts = bl.initialize_app(force_regenerate=args.regenerate)

    app.run(debug=True, host='0.0.0.0', port=5000)