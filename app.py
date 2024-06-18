from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Carregar o modelo
model = tf.keras.models.load_model('model.h5')

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    reshaped = resized.reshape(1, 28, 28, 1) / 255.0
    return reshaped

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

if __name__ == "__main__":
    app.run(debug=True)
