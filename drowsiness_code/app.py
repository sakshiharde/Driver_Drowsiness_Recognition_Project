from flask import Flask, render_template, Response
import cv2
import numpy as np
from inference import DrowsinessDetector
import os

app = Flask(__name__)

# Load the TFLite model using your class
detector = DrowsinessDetector("model.tflite")

# Video stream generator
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Get prediction from model
        pred = detector.predict(frame)  # returns 0 or 1
        label = "Awake 🙂" if pred == 0 else "Drowsy 😴"
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)

        # Draw a box and label
        h, w, _ = frame.shape
        start_point = (100, 100)
        end_point = (w - 100, h - 100)
        cv2.rectangle(frame, start_point, end_point, color, 2)
        cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)

    

