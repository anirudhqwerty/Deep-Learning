# prediction.py - UPDATED

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Use the SAME model architecture as training
class ImprovedCatDogCNN(nn.Module):
    def __init__(self):
        super(ImprovedCatDogCNN, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Second block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Third block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Fourth block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x

# Load trained model
model = ImprovedCatDogCNN().to(device)

# Handle both checkpoint formats
try:
    checkpoint = torch.load("best_catdog_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded best model")
except:
    checkpoint = torch.load("catdog_cnn.pth", map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Loaded fallback model")

model.eval()

# Use the SAME transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ask user for image path
image_path = input("Enter the path to your image: ")

try:
    image = Image.open(image_path).convert("RGB")
except:
    print("Error: Could not open image. Check the path.")
    exit()

image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    pred = output.item()
    
    print(f"Raw output: {pred}")
    
    # Correct logic for your model
    if pred < 0.5:
        print("Prediction: Cat 🐱")
        print(f"Confidence: {(1-pred):.1%}")
    else:
        print("Prediction: Dog 🐶")
        print(f"Confidence: {pred:.1%}")
