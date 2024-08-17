import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import os
from sklearn.cluster import KMeans

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_eye_color(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        return "No eyes detected"

    (x, y, w, h) = eyes[0]
    eye_image = image[y:y+h, x:x+w]

    eye_image_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    pixels = eye_image_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    dominant_color = colors[unique[np.argmax(counts)]]

    color_name = get_color_name(dominant_color)
    return color_name

def get_color_name(rgb):
    color_map = {
        'Brown': (42, 42, 42),
        'Blue': (0, 0, 255),
        'Green': (0, 255, 0),
        'Gray': (0, 0, 128),
        'Black': (128, 128, 128),
        'Light Blue': (173, 216, 230),
        'Dark Brown': (101, 67, 33)
    }
    
    min_distance = float('inf')
    closest_color = 'Unknown'
    
    for color, value in color_map.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(value))
        if distance < min_distance:
            min_distance = distance
            closest_color = color
            
    return closest_color

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = detect_eye_color(filepath)
            return render_template('result.html', result=result, filename=filename)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
