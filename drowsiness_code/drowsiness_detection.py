import tensorflow as tf
import os

model_path = os.path.join("Driver_Drowsiness", "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load the TFLite model
#interpreter = tf.lite.Interpreter(model_path="model.tflite")
#import os
#print("Working directory:", os.getcwd())
#print("Files here:", os.listdir())



