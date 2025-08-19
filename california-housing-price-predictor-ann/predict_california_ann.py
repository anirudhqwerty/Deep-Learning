import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

housing = fetch_california_housing()
feature_names = housing.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(housing.data)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

input_size = X_scaled.shape[1]
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

#loading trained model from train_california_ann.py
model.load_state_dict(torch.load("california_ann_model.pth", map_location=device))
model.eval()

print("Enter details about the house : ")

user_input = []
for feature in feature_names:
    while True:
        try:
            val = float(input(f"{feature}: "))
            user_input.append(val)
            break
        except:
            print("Please enter a valid number.")

user_input_np = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input_np)
user_tensor = torch.tensor(user_input_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    predicted_price = model(user_tensor).item()

print(f"\nPredicted California House Price: ${predicted_price*100000:.2f}")
