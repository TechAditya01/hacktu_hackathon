import joblib
from architecture import MLP
import torch
import pandas as pd
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Load the model
# model = MLP(input_dim=26, num_classes=2)
# model.load_state_dict(torch.load('loan_approval_model.pth'))
# model.eval()  # Set the model to evaluation mode
model = joblib.load('model.joblib')

# Load the preprocessor
preprocessor = joblib.load('loan_approval_preprocessor.joblib')

# Load label encoders
label_encoders = joblib.load('label_encoders.joblib')

def get_JSON_input(example_input):
    with open(example_input, 'r') as f:
        user_data = json.load(f)
    return user_data

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

        # Preprocess input data
        user_tensor = preprocess_input(data)

        # Make prediction
        with torch.no_grad():
            prediction = model(user_tensor)
            probabilities = torch.sigmoid(prediction)
            predicted_class = (probabilities > 0.5).int()
            # Get the predicted target by selecting the index with the highest logit
            # print("Prediction:", prediction)
            predicted_class = torch.argmax(prediction, dim=1)
            # print("Predicted class:", predicted_class.item())
            if predicted_class.item() == 0:
                result = "The model predicts that the loan will not be approved."
            else:
                result = "The model predicts that the loan will be approved."

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
