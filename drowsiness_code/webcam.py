import cv2
from inference import DrowsinessDetector

detector = DrowsinessDetector("model.tflite")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    prediction = detector.predict(frame)
    print("Prediction:", prediction)

    cv2.imshow("Webcam - Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
