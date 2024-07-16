import pandas as pd
import numpy as np
import pickle
import joblib  # Import joblib for loading LabelEncoder
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the machine learning model
with open('CO2(1).pickle', 'rb') as handle:
    model = pickle.load(handle)

# Load the LabelEncoders (assuming you have separate encoders for each categorical feature)
labelencoder_country = joblib.load('labelencoder_country.joblib')
labelencoder_code = joblib.load('labelencoder_code.joblib')
labelencoder_indicator = joblib.load('labelencoder_indicator.joblib')

@app.route('/')
def home():
    return render_template('first.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('second.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('first.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    try:
        # Extract input features from the form
        country_name = request.form['cname']
        country_code = request.form['Ccode']
        indicator_name = request.form['Iname']
        year = float(request.form['year'])  # Convert year to float
        
        # Transform categorical variables using LabelEncoders
        country_name_encoded = labelencoder_country.transform([country_name])[0]
        country_code_encoded = labelencoder_code.transform([country_code])[0]
        indicator_name_encoded = labelencoder_indicator.transform([indicator_name])[0]
        
        # Create a DataFrame with the input features
        features_values = [[country_name_encoded, country_code_encoded, indicator_name_encoded, year]]
        feature_names = ['CountryName', 'CountryCode', 'IndicatorName', 'Year']
        x = pd.DataFrame(features_values, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(x)
        print("Prediction is:", prediction)
        
        # Ensure prediction is not empty or None
        if prediction is not None and len(prediction) > 0:
            # Format the prediction value to desired format
            formatted_prediction = "{:.11f}".format(prediction[0])  # Adjust precision as needed
            
            # Render result template with the formatted prediction
            return render_template("result.html", prediction=formatted_prediction)
        else:
            return "Prediction could not be made."
    
    except Exception as e:
        print("Error:", str(e))
        return str(e)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
