import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan scaler
model = joblib.load('D:/KULIAH/TELKOM_UNIVERSITY/SEMESTER_8/CITRA_DIGITAL/TUBES/KODE_Train/Aplikasi/model/voting_model_baru.pkl')
scaler = joblib.load('D:/KULIAH/TELKOM_UNIVERSITY/SEMESTER_8/CITRA_DIGITAL/TUBES/KODE_Train/Aplikasi/model/scaler_baru.pkl')
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Ekstraksi fitur (sama dengan training)
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb_hist = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256]).flatten()
        rgb_hist.extend(hist)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))

    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]

    sobel_img = sobel(gray)
    sobel_hist, _ = np.histogram(sobel_img.ravel(), bins=16, range=(0, 1))

    return np.concatenate([rgb_hist, hue_hist, lbp_hist, glcm_props, sobel_hist])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            features = extract_features(filepath).reshape(1, -1)
            features_scaled = scaler.transform(features)
            proba = model.predict_proba(features_scaled)[0]
            pred_index = np.argmax(proba)
            pred_label = model.classes_[pred_index]
            confidence = round(proba[pred_index] * 100, 2)

            return render_template("index.html", label=pred_label, confidence=confidence, image_path=filepath)

            #pred = model.predict(features_scaled)[0]
            #return render_template("index.html", label=pred, image_path=filepath)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
