from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [[
        float(data['n']),
        float(data['p']),
        float(data['k']),
        float(data['temperature']),
        float(data['humidity']),
        float(data['ph']),
        float(data['rainfall'])
    ]]

    prediction = model.predict(features)

    return jsonify({
        "prediction": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)