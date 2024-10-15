from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

# Load the model
model = pickle.load(open('car_price_model.pkl', 'rb'))

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form.get('name')
        company = request.form.get('company')
        fuel_type = request.form.get('fuel_type')
        year = request.form.get('year')
        kms_driven = request.form.get('kms_driven')

        # Validate and convert inputs
        if not year or not kms_driven:
            return jsonify({"error": "Year and KMs Driven are required"}), 400

        year = float(year)
        kms_driven = float(kms_driven)

        # Create DataFrame from input
        input_query = pd.DataFrame([[name, company, fuel_type, year, kms_driven]],
                                   columns=['name', 'company', 'fuel_type', 'year', 'kms_driven'])

        # Predict using the model
        result = model.predict(input_query)[0]

        # Return the prediction
        return jsonify({"price": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400  # Handle conversion errors
    except Exception as e:
        # Handle other errors
        return jsonify({"error": "An error occurred during prediction: " + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
