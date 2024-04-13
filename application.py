import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder='template')

# Load the trained model
with open('LinearRegressionModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the DataFrame
df = pd.read_csv("Players-Data.csv")

@app.route('/')
def index():
    Overall = sorted(df['Overall'].unique())
    return render_template('index.html', Overall=Overall)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the overall rating from the form
    Overall = int(request.form.get('Overall'))

    # Make prediction
    prediction = model.predict([[Overall]])

    # Return the prediction as a string
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
