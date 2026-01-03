import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Disable OneDNN & Reduce TensorFlow Logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# Load Tokenizer
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# Load TFLite model
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="fake_job_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")

MAX_SEQUENCE_LENGTH = 200

def preprocess_text(text):
    # Optimize: Truncate very long text to avoid processing unnecessary tokens.
    # The model uses the last 200 tokens (pad_sequences default truncating='pre').
    # We keep a safe buffer (e.g., 1000 words) to ensure we don't lose relevant tokens.
    words = text.split()
    if len(words) > MAX_SEQUENCE_LENGTH * 5:
        text = " ".join(words[-MAX_SEQUENCE_LENGTH * 5:])

    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    combined_text = request.form.get("combined_text")

    if not combined_text:
        return render_template("index.html", prediction="Please enter the job description.")

    # Preprocess the input
    input_data = preprocess_text(combined_text)

    # Run inference using TFLite model
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    # Determine result
    result = "Fraudulent" if prediction > 0.7 else "Legitimate"

    return render_template("index.html", prediction=f"The job post is {result}")

if __name__ == "__main__":
    app.run(debug=True)



