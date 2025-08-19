import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


housing = fetch_california_housing()
X = housing.data
y = housing.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


input_size = X_scaled.shape[1]
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)



model.load_state_dict(torch.load("california_ann_model.pth", map_location=device))
model.eval()




with torch.no_grad():
    predictions = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()


mse = mean_squared_error(y_true, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
