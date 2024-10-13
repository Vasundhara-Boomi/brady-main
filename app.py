from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from bplot import BPlot

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Mapping of filenames to video URLs
mapping = {
    'severe.MOV': ['AlphaPose_severe.mp4', 'alphapose-results-severe (1).json'],
    'mild.MOV': ['AlphaPose_mild.mp4', 'alphapose-results-mild.json'],
    'moderate.MOV': ['AlphaPose_moderate.mp4', 'alphapose-results-moderate.json'],
    'slight.MOV': ['AlphaPose_slight.mp4', 'alphapose-results-slight.json'],
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
        
        # Check if the uploaded filename exists in the mapping
        if filename in mapping:
            video_url = mapping[filename][0]  # Get the mapped video URL
            return jsonify({'success': True, 'video_url': url_for('video_display', video_url=video_url, videotype=filename)})
        else:
            return jsonify({'success': False, 'error': 'File not found in mapping'}), 404
    return jsonify({'success': False, 'error': 'Invalid request'}), 400

@app.route('/video_display')
def video_display():
    videotype = request.args.get('videotype')
    video_url = request.args.get('video_url')  # Get the video URL
    print(f"Video URL for display: {video_url}")  # Debug statement

    plotter = BPlot(videotype[:-4])
    plotter.convert()
    
    # Plot and return base64 encoded image for HTML
    graph_image = plotter.plot_amplitude()
    distance_image = plotter.plot_speed()

    severity = plotter.determine_severity()
    return render_template('video_display.html', video=video_url,graph_image=graph_image, distance_image=distance_image, score=severity)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, port=8000)