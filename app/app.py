# --- Importing to use BLL, web and arguments ---
from flask import Flask, render_template, request
import business_logic as bl
import argparse
import os


# --- Global Artifacts ---
# These are loaded only once when the application starts!
artifacts = None


# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')


# Initialize the Flask application, specifying the correct template and static folder paths
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


# --- Web Routes ---
@app.route('/')
def index():
    # It will look for 'index.html' in the specified TEMPLATE_DIR
    return render_template('index.html')


@app.route('/search')
def search():
    # 1. Read search parameters from the URL (e.g., /search?artist=...&song=...)
    artist_name = request.args.get('artist')
    song_name = request.args.get('song')

    results = None
    if artist_name and song_name:
        # 2. Call the get_recommendations function from 'business_logic.py'
        results = bl.get_recommendations(
            artist_name,
            song_name, 
            artifacts["lyrics_df"], 
            artifacts["tfidf_matrix"]
        )
    
    # 3. Pass the results to the 'results.html' template for rendering
    return render_template(
        'results.html', 
        artist=artist_name,
        song=song_name,
        results=results
    )

# --- NEW: 404 Error Handler ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# --- Application Startpoint ---
if __name__ == '__main__':
    # Parse command-line arguments (e.g., --regenerate)
    parser = argparse.ArgumentParser(description="Music Mood Classifier Web App")
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="Force regeneration of all models and processed data on start."
    )
    args = parser.parse_args()

    # 1. Load all models and data into memory ON STARTUP
    print("Flask app starting...")
    print("Initializing business logic (this may take a few minutes)...")
    try:
        # Pass the 'force_regenerate' flag to the initializer
        artifacts = bl.initialize_app(force_regenerate=args.regenerate)
        print("Initialization complete. Server is running.")
        
        # 2. Run the web server
        # 'debug=True' helps with development (auto-reloads on save).
        # This should be set to 'False' in a production environment.
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except RuntimeError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Application could not start.")