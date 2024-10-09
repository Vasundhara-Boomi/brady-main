from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mapping of filenames to video URLs
mapping = {
    'severe.MOV': [r'D:\7thSem\finger-tapping-severity-main\fingertapping\gcn\recorded\originals\AlphaPose_severe.mp4', 'alphapose-results-severe (1).json'],
    'mild.MOV': ['AlphaPose_mild.mp4', 'alphapose-results-mild (1).json'],
    'moderate.MOV': ['AlphaPose_moderate.mp4', 'alphapose-results-moderate (1).json'],
    'slight.MOV': ['AlphaPose_slight.mp4', 'alphapose-results-slight (1).json'],
    'normal.MOV': ['AlphaPose_normal.mp4', 'alphapose-results-normal.json']
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload_files():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        print(f"Uploading file: {filename}")  # Debug
        return redirect(url_for('upload', filename=filename))
    return 'Invalid request'

@app.route('/uploads/<filename>', methods=["GET"])
def upload(filename):
    
    mapped_videos = mapping.get(filename, [])
    filename = filename[:-4]  # Remove the file extension for mapping lookup
    
    video_url = mapped_videos[0]  # Get the URL for the video
    print("Video URL:", video_url)  # Debug: print the video URL
    return render_template('video_display.html', videos=video_url)  # Render the template with the video URL
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, port=8000)
