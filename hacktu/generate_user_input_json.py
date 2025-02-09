import json

def gather_inputs():
    # ...existing code for interactive input as needed...
    # For brevity, using fixed sample values matching example_input.json
    user_data = {
        "Age": 30,
        "AnnualIncome": 60000,
        "CreditScore": 700,
        "EmploymentStatus": "employed",
        "EducationLevel": "bachelor",
        "Experience": 10,
        "LoanAmount": 20000,
        "LoanDuration": 36,
        "MaritalStatus": "married",
        "NumberOfDependents": 0,
        "HomeOwnershipStatus": "own",
        "MonthlyDebtPayments": 500,
        "LoanPurpose": "auto",
        "SavingsAccountBalance": 10000,
        "CheckingAccountBalance": 5000,
        "MonthlyIncome": 4000,
        "JobTenure": 5,
        "NetWorth": 150000,
        "BaseInterestRate": 3.5,
        "InterestRate": 5.0,
        "MonthlyLoanPayment": 600
    }
    return user_data

if __name__ == "__main__":
    inputs = gather_inputs()
    with open('example_input.json', 'w') as f:
        json.dump(inputs, f, indent=2)
    print("User input data has been saved to example_input.json")
