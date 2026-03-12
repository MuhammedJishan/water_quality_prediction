from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1,-1)
    features = scaler.transform(features)
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Water is Safe to Drink"
    else:
        result = "Water is Not Safe to Drink"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run()