import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [
        float(request.form['Nitrogen']),
        float(request.form['Phosphorus']),
        float(request.form['Potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['PH']),
        float(request.form['rainfall'])
    ]
    features = np.array([float_features])
    prediction = model.predict(features)[0]
    return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction}")

if __name__ == "__main__":
    flask_app.run(debug=True)
