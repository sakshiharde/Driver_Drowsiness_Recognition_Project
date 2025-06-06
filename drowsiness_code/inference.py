
import os
import numpy as np
import tensorflow as tf
import cv2

class DrowsinessDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "model.tflite")
        else:
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def predict(self, frame):
        # Resize and preprocess
        input_frame = cv2.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return int(np.argmax(output_data))
