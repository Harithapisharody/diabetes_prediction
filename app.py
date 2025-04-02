from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models
diabetes_model_path = r'C:\Users\HP\Documents\Haritha\Honor_project\final\new\best_diabetes_model.pkl'
glucose_model_path = r'C:\Users\HP\Documents\Haritha\Honor_project\final\new\glucose_forecast_model.pkl'

diabetes_model = joblib.load(diabetes_model_path)
glucose_model = joblib.load(glucose_model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Check for missing values
        if any(value == "" or value is None for value in data.values()):
            return jsonify({"error": "All input fields are required"}), 400
        
        # Diabetes prediction
        diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        diabetes_input = pd.DataFrame([data], columns=diabetes_features)
        diabetes_prediction = diabetes_model.predict(diabetes_input)[0]  # Fixing decimal issue
        diabetes_result = "Diabetic Please Consult Doctor" if diabetes_prediction ==1 else "Non-Diabetic"

        return jsonify({
            "diabetes_prediction": diabetes_result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/glucose-predict', methods=['POST'])
def glucose_predict():
    try:
        data = request.get_json()
        
        # Check for missing values
        if any(value == "" or value is None for value in data.values()):
            return jsonify({"error": "All input fields are required"}), 400
        
        # Glucose prediction
        glucose_features = ["Age", "BMI", "Insulin", "BloodPressure"]
        glucose_input = pd.DataFrame([data], columns=glucose_features)
        glucose_prediction = glucose_model.predict(glucose_input)[0] / 10

        return jsonify({
            "predicted_glucose": round(glucose_prediction, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
