import joblib
from architecture import MLP
import torch
import pandas as pd
from flask import Flask, request, jsonify
import json
import os
import psycopg2  # Import the psycopg2 library

app = Flask(__name__)

# Database configuration from environment variables
DB_HOST = os.environ.get("DB_HOST", "localhost")  # Default to localhost
DB_NAME = os.environ.get("DB_NAME", "loan_db")    # Replace with your DB name
DB_USER = os.environ.get("DB_USER", "loan_user")    # Replace with your username
DB_PASS = os.environ.get("DB_PASS", "loan_password") # Replace with your password
DB_PORT = os.environ.get("DB_PORT", "5432") # Replace with your port

# Load the model
model = joblib.load('model.joblib')

# Load the preprocessor
preprocessor = joblib.load('loan_approval_preprocessor.joblib')

# Load label encoders
label_encoders = joblib.load('label_encoders.joblib')

def get_user_data_from_db(user_id):
    try:
        # Establish database connection
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()

        # Fetch user data from the database
        cur.execute("SELECT Age, AnnualIncome, CreditScore, EmploymentStatus, EducationLevel, Experience, LoanAmount, LoanDuration, MaritalStatus, NumberOfDependents, HomeOwnershipStatus, MonthlyDebtPayments, LoanPurpose, SavingsAccountBalance, CheckingAccountBalance, MonthlyIncome, JobTenure, NetWorth, BaseInterestRate, InterestRate, MonthlyLoanPayment FROM loan_application WHERE id = %s;", (user_id,))  # Replace 'users' with your table name
        user_data = cur.fetchone()

        if user_data:
            # Convert the tuple to a dictionary, assuming the order of columns
            # matches the order expected by your preprocessing
            column_names = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'Experience', 'LoanAmount', 'LoanDuration', 'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments', 'LoanPurpose', 'SavingsAccountBalance', 'CheckingAccountBalance', 'MonthlyIncome', 'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment']
            user_data = dict(zip(column_names, user_data))
        else:
            return None

        cur.close()
        conn.close()
        return user_data

    except Exception as e:
        print(f"Database error: {e}")
        return None

def preprocess_input(user_data):
    numeric_cols = [
        'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount',
        'LoanDuration', 'MonthlyDebtPayments',
         'SavingsAccountBalance', 'CheckingAccountBalance', 'MonthlyIncome', 'JobTenure', 'NetWorth',
        'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
    ]
    cat_cols = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'LoanPurpose', 'HomeOwnershipStatus']
    
    # Process categorical inputs using LabelEncoder(s)
    for col in cat_cols:
        user_data[col] = int(label_encoders[col].transform([user_data[col]])[0])
    
    # ...existing feature engineering code...
    user_data['TotalIncome'] = user_data['AnnualIncome'] + user_data['SavingsAccountBalance'] + user_data['CheckingAccountBalance']
    user_data['DebtToIncomeRatio'] = user_data['MonthlyDebtPayments'] / (user_data['MonthlyIncome'] + 1e-5)
    user_data['CreditScore_Income'] = user_data['CreditScore'] * user_data['AnnualIncome']
    user_data['DebtToIncome_CreditScore'] = user_data['DebtToIncomeRatio'] * user_data['CreditScore']
    user_data['InterestRate_LoanDuration'] = user_data['InterestRate'] * user_data['LoanDuration']

    user_df = pd.DataFrame([user_data])
    # Rearrange columns to match training order
    list_order = ['Age',
    'AnnualIncome',
    'CreditScore',
    'EmploymentStatus',
    'EducationLevel',
    'Experience',
    'LoanAmount',
    'LoanDuration',
    'MaritalStatus',
    'NumberOfDependents',
    'HomeOwnershipStatus',
    'MonthlyDebtPayments',
    'DebtToIncomeRatio',
    'LoanPurpose',
    'SavingsAccountBalance',
    'CheckingAccountBalance',
    'MonthlyIncome',
    'JobTenure',
    'NetWorth',
    'BaseInterestRate',
    'InterestRate',
    'MonthlyLoanPayment',
    'TotalIncome',
    'CreditScore_Income',
    'DebtToIncome_CreditScore',
    'InterestRate_LoanDuration']
    user_df = user_df.reindex(columns=list_order)
    
    # Check if any column contains NaN values and raise an error if so
    if user_df.isnull().any().any():
        raise ValueError("Input data contains NaN values")
    
    user_preprocessed = preprocessor.transform(user_df)
    
    # Convert to PyTorch tensor
    user_tensor = torch.tensor(user_preprocessed.tolist(), dtype=torch.float32)
    return user_tensor

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        user_id = data.get('user_id')  # Expect a 'user_id' in the request

        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        # Fetch user data from the database
        user_data = get_user_data_from_db(user_id)
        if not user_data:
            return jsonify({'error': 'User not found'}), 404
        
        # Preprocess input data
        user_tensor = preprocess_input(user_data)

        # Make prediction
        with torch.no_grad():
            model.eval()
            prediction = model(user_tensor)
            # probabilities = torch.sigmoid(prediction)
            # predicted_class = (probabilities > 0.5).int()
            # Get the predicted target by selecting the index with the highest logit
            # print("Prediction:", prediction)
            predicted_class = torch.argmax(prediction, dim=1)
            # print("Predicted class:", predicted_class.item())
            if predicted_class.item() == 0:
                result = "The model predicts that the loan will not be approved."
            else:
                result = "The model predicts that the loan will be approved."

        return jsonify({'prediction': predicted_class.item()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
