import torch
# Detectar el dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

#1. Importamos el dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("puneet6060/intel-image-classification")

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 2. Definimos las transformaciones para normalizar los datos
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),  # Convertimos imágenes a Tensores
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Cargar dataset
train_data = datasets.ImageFolder(root=f"{path}/seg_train/seg_train", transform=transform)
test_data = datasets.ImageFolder(root=f"{path}/seg_test/seg_test", transform=transform)

train_loader = DataLoader(train_data, batch_size=75, shuffle=True)
test_loader = DataLoader(test_data, batch_size=75, shuffle=False)

# 4. Mostrar clases
print(f"Clases encontradas: {train_data.classes}")

# 5. Vamos a visualizar una imagen del dataset
import matplotlib.pyplot as plt
import numpy as np

# Obtenemos una imagen del dataset
i = np.random.randint(len(train_data))
image, label = train_data[i]
image = image.unsqueeze(0).to(device)

plt.imshow(image.squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.show()

import torch.nn as nn
import torch.nn.functional as F

#5. Definimos el modelo
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)  # Entrada: 3 canales, Salida: 32 canales, kernel de 5x5
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling con un kernel de 2x2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Entrada: 32 canales, Salida: 64 canales
        self.fc1 = nn.Linear(64 * 37 * 37, 256)  # Imágenes reducidas a 150x150 después del pooling
        self.fc2 = nn.Linear(256, 6)  # 100 clases de salida para MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)  # Aplanar
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet().to(device)
print(model)
print("Número de parámetros:", sum(p.numel() for p in model.parameters()))

# 6. Predicción antes del entrenamiento
model.eval()
with torch.no_grad():
    output = model(image)  # Agregar dimensión batch
    _, pred = torch.max(output, 1)  # Índice de la mayor probabilidad

# Mostrar resultados
print(f"Etiqueta real: {train_data.classes[label]}")
print(f"Predicción del modelo: {train_data.classes[pred]}")

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#6. Entrenamiento
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Limpiamos los gradientes
        outputs = model(images)  # Pasamos las imágenes por la red
        loss = criterion(outputs, labels)  # Calculamos la pérdida
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizamos los pesos

        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    #7. EValuación
correct = 0
total = 0
with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        #predicción despues del entrenamiento
        output = model(image)  # Agregar dimensión batch
        _,pred = torch.max(output, 1)  # Índice de la mayor probabilidad

print(f'Accuracy of the model on the test images: {100 * correct / total}%')
# Mostrar resultados
print(f"Etiqueta real: {train_data.classes[label]}")
print(f"Predicción del modelo: {train_data.classes[pred]}")