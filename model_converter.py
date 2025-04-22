# import tensorflow.lite as tflite

# interpreter = tflite.Interpreter(model_path="sign_language_model.tflite")
# interpreter.allocate_tensors()

# print("TFLite model loaded successfully!")



import tensorflow as tf

try:
    model = tf.keras.models.load_model(r'C:\Users\Karthik\Desktop\project_testing_2\sign_language_modelll.h5')
    print("Model loaded successfully!")
    print(f"Model summary: {model.summary()}")
except Exception as e:
    print(f"Error loading model: {e}")

