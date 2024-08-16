from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model('models/ann_model.h5')
scaler = joblib.load('models/scaler.lb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract data from form and convert to appropriate types
    try:
        name = request.form.get('name', 'Unknown')
        no_of_dependents = float(request.form.get('no_of_dependents', 0))
        income_annum = float(request.form.get('income_annum', 0))
        loan_amount = float(request.form.get('loanAmount', 0))
        loan_term = float(request.form.get('loan_term', 0))
        cibil_score = float(request.form.get('cibil_score', 0))
        residential_assets_value = float(request.form.get('residential_assets_value', 0))
        commercial_assets_value = float(request.form.get('commercial_assets_value', 0))
        luxury_assets_value = float(request.form.get('luxury_assets_value', 0))
        bank_asset_value = float(request.form.get('bank_asset_value', 0))
        
        education = request.form.get('education', 'Not Graduate')
        self_employed = request.form.get('self_employed', 'No')
        
        education_numeric = 1 if education == 'Graduate' else 0
        self_employed_numeric = 1 if self_employed == 'Yes' else 0
        
        input_data = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                                residential_assets_value, commercial_assets_value, luxury_assets_value,
                                bank_asset_value, education_numeric, self_employed_numeric]])

        input_transformed = scaler.transform(input_data)
        prediction = model.predict(input_transformed)

        pred = (prediction > 0.5).astype(int)
        value = int(pred[0][0])

        if value == 0:
            result = 'Approved'
            img_file = 'loan-approved.jpg'
        else:
            result = 'Rejected'
            img_file = 'loan- rejected.jpg'

        
    
    except Exception as e:
        # Handle errors
        result = f"Error: {str(e)}"
    
    return render_template('output.html', output=result, img_file=img_file)

if __name__ == "__main__":
    app.run(debug=True)
