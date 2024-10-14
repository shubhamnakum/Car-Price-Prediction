from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging
model = pickle.load(open('car_price_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data
        name = data.get('name')
        company = data.get('company')
        fuel_type = data.get('fuel_type')
        year = data.get('year')
        kms_driven = data.get('kms_driven')

        input_query = pd.DataFrame([[name, company, fuel_type, year, kms_driven]], columns=[
                                   'name', 'company', 'fuel_type', 'year', 'kms_driven'])
        result = model.predict(input_query)[0]

        # Return a JSON response with the predicted price
        return jsonify(price=result)
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
