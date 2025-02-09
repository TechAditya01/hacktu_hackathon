import joblib
import torch
import pandas as pd

# Load the model saved using joblib
model = joblib.load('model.joblib')

# Prepare example input data (adjust columns/order as required)
# Here we create a dummy dataframe with 26 features
data = pd.DataFrame([[0.5] * 26], columns=[f'feature_{i}' for i in range(26)])

# Convert input data to a PyTorch tensor
input_tensor = torch.tensor(data.values, dtype=torch.float32)

# Set the model to evaluation mode and make prediction
model.eval()
with torch.no_grad():
    prediction = model(input_tensor)
    print("Model prediction:", prediction)
