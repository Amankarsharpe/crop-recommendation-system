
from flask import Flask, render_template, request
import numpy as np
import pickle
import webbrowser

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model
with open("crop_model_1.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")


# Predict page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input values from form
            N = float(request.form.get("N", 0))
            P = float(request.form.get("P", 0))
            K = float(request.form.get("K", 0))
            temperature = float(request.form.get("temperature", 0))
            humidity = float(request.form.get("humidity", 0))
            ph = float(request.form.get("ph", 0))
            rainfall = float(request.form.get("rainfall", 0))

            # Prepare data for prediction
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            return render_template("result.html", crop=prediction)
        except Exception as e:
            return f"Error: {e}"

    return render_template("predict.html")


if __name__ == "__main__":
    # Open default browser automatically
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
