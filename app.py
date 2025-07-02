# File: app.py (Versi Stateless API)

from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import base64
import io
from PIL import Image

app = Flask(__name__, template_folder='views')

# --- FUNGSI HELPER (Tidak ada perubahan) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3]); ear = (A + B) / (2.0 * C); return ear
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10]); B = dist.euclidean(mouth[4], mouth[8]); C = dist.euclidean(mouth[0], mouth[6]); mar = (A + B) / (2.0 * C); return mar

# --- KONSTANTA & INISIALISASI MODEL (Tidak ada perubahan) ---
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.45
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42, 48); (rStart, rEnd) = (36, 42); (mStart, mEnd) = (48, 68)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    
    pil_image = Image.open(io.BytesIO(image_data))
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    is_eye_closed = False
    is_mouth_open = False
    
    rects = detector(gray, 0)
    if len(rects) > 0:
        shape = predictor(gray, rects[0]) # Hanya proses satu wajah
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])

        if ear < EAR_THRESHOLD:
            is_eye_closed = True
        
        if mar > MAR_THRESHOLD:
            is_mouth_open = True

    # Kembalikan analisis HANYA untuk frame ini
    return jsonify({
        'is_eye_closed': is_eye_closed,
        'is_mouth_open': is_mouth_open
    })

if __name__ == '__main__':
    app.run(debug=True)