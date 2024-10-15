from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('car_price_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    company = request.form.get('company')
    fuel_type = request.form.get('fuel_type')
    year = request.form.get('year')
    kms_driven = request.form.get('kms_driven')

    input_query = pd.DataFrame([[name, company, fuel_type, year, kms_driven]], columns=[
                               'name', 'company', 'fuel_type', 'year', 'kms_driven'])
    result = model.predict(input_query)

    return jsonify({"price": result})


if __name__ == '__main__':
    app.run(debug=True)
