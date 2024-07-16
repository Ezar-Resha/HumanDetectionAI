from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from ultralytics import YOLO
from model import detect_people, models, load_model, process_image, process_video
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    return render_template('index.html', models=[m['name'] for m in models])


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = process_video(file_path)

        return render_template('result_video.html', results=results)


@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Determine the file type by its extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            # Process image
            original_image_path, results = process_image(file_path)
            return render_template('result.html', original_image_path=original_image_path, results=results)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            # Process video
            results = process_video(file_path)
            return render_template('result_video.html', results=results)
        else:
            # Unsupported file type
            return "Unsupported file type", 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))


@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True)
