import joblib
from architecture import MLP
import torch
import pandas as pd
import json

# loading model 
model = MLP(input_dim=26, num_classes=2)
model.load_state_dict(torch.load('loan_approval_model.pth'))

def get_user_inputs():
    # Define which columns are numeric and which are categorical
    numeric_cols = [
        'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount',
        'LoanDuration', 'MonthlyDebtPayments',
         'SavingsAccountBalance', 'CheckingAccountBalance', 'MonthlyIncome', 'JobTenure', 'NetWorth',
        'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
    ]
    cat_cols = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'LoanPurpose', 'HomeOwnershipStatus']
    
    
    # user_data = {}
    # Collect numeric inputs
    # for col in numeric_cols:
    #     val = input(f"Enter numeric value for {col}: ")
    #     try:
    #         user_data[col] = float(val)
    #     except ValueError:
    #         print(f"Invalid numeric input for {col}.")
    #         return None
    # # Increase CreditScore weight for user input (match training)
    # user_data['CreditScore'] = user_data['CreditScore'] * 2
    
    # Load pre-fitted label encoders for categorical columns
    label_encoders = joblib.load('label_encoders.joblib')
    
    # Process categorical inputs using LabelEncoder(s)
    # for col in cat_cols:
    #     options = list(label_encoders[col].classes_)
    #     print(f"Options for {col}: {options}")
    #     val = input(f"Enter category for {col}: ")
    #     try:
    #         user_data[col] = int(label_encoders[col].transform([val])[0])
    #     except Exception:
    #         print(f"Invalid input for {col}.")
    #         return None
        
    # if JSON input is used, the following code can be used to load the user data
    def get_JSON_input(example_input):
        with open(example_input, 'r') as f:
            user_data = json.load(f)
            
        # use label encoders to transform categorical columns
        for col in cat_cols:
            user_data[col] = int(label_encoders[col].transform([user_data[col]])[0])
        return user_data
    
    user_data = get_JSON_input('example_input.json')
    
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
    
    # Load preprocessor and apply transformation
    preprocessor = joblib.load('loan_approval_preprocessor.joblib')
    user_preprocessed = preprocessor.transform(user_df)
    
    # Convert to PyTorch tensor
    user_tensor = torch.tensor(user_preprocessed.tolist(), dtype=torch.float32)
    return user_tensor

# Example usage:
user_tensor = get_user_inputs()
if user_tensor is not None:
    with torch.no_grad():
        model.eval()
        prediction = model(user_tensor)
        # Get the predicted target by selecting the index with the highest logit
        # print("Prediction:", prediction)
        predicted_class = torch.argmax(prediction, dim=1)
        # print("Predicted class:", predicted_class.item())
        if predicted_class.item() == 0:
            print("The model predicts that the loan will not be approved.")
        else:
            print("The model predicts that the loan will be approved.")