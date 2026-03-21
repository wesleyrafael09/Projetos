
Utilizando Modelo Pré Treinado 
https://colab.research.google.com/drive/1TgGzoUl4UWwcBWVNfGkXpOfAGgpJY1Sv

Utilizando Modelo criado do zero uma CNN simples 
https://colab.research.google.com/drive/124_v0M06lR90Log6ExuRRZIocbsBjOuZ




# Solução Apresentada utilizando modelo pre-treinado Resnet18
# Instalar Dependências
!pip install torch torchvision scikit-learn

# Importações
import os
import zipfile
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import classification_report, confusion_matrix

# Configurações de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Usando dispositivo: {device}")

# Definir URLs
background_url = "https://www.dropbox.com/scl/fo/myx9nxbavqi073ll365qo/AJWKN4UAPvlIa5n8ZUHWST4/Background?rlkey=3h0kub8j36rthlwos741vh78h&dl=1"
pedestrian_url = "https://www.dropbox.com/scl/fo/myx9nxbavqi073ll365qo/AAPMm6XJI96pJd5QXC08toA/Pedestrians?rlkey=3h0kub8j36rthlwos741vh78h&dl=1"

# Download e Preparação do Dataset
os.makedirs("data_pedestrian", exist_ok=True)

# Baixar arquivos
!wget -O background.zip "$background_url"
!wget -O pedestrian.zip "$pedestrian_url"

# Descompactar
with zipfile.ZipFile("background.zip", "r") as zip_ref:
    zip_ref.extractall("data_pedestrian/Background")

with zipfile.ZipFile("pedestrian.zip", "r") as zip_ref:
    zip_ref.extractall("data_pedestrian/Pedestrians")


    # Organizar estrutura
dataset_path = "data_pedestrian/dataset"
os.makedirs(dataset_path, exist_ok=True)

shutil.move("data_pedestrian/Background", os.path.join(dataset_path, "background"))
shutil.move("data_pedestrian/Pedestrians", os.path.join(dataset_path, "pedestrian"))

print(" Dataset preparado em:", dataset_path)


# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



# Carregar Dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print(f" Total de imagens: {len(full_dataset)}")
print(f" Classes: {full_dataset.classes}")



# Divisão treino, val, teste
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f" Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Modelo ResNet18
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet18.parameters():
    param.requires_grad = False



    # Substituir a última camada
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 2)
resnet18 = resnet18.to(device)
print(resnet18)

# Funções auxiliares
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.fc.parameters(), lr=0.001)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


    #Loop de Treinamento
import os
import torch

# Configuração
num_epochs = 10
modelo_path = 'modelos/melhor_modelo_resnet18.pth'

# Cria a pasta (se não existir)
os.makedirs('modelos', exist_ok=True)

best_val_acc = 0
best_model_wts = None  # Guarda os melhores pesos com base na validação

print(f"{'Época':>7} {'Train Loss':>12} {'Train Acc':>12} | {'Val Loss':>10} {'Val Acc':>10}")
print("-" * 65)

for epoch in range(num_epochs):
    print(f"Época {epoch+1}/{num_epochs}")

    # Treinamento
    resnet18.train()
    train_loss, train_acc = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_acc += accuracy(outputs, labels) * inputs.size(0)
    train_loss /= train_size
    train_acc /= train_size

    # Validação
    resnet18.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_acc += accuracy(outputs, labels) * inputs.size(0)
    val_loss /= val_size
    val_acc /= val_size

    print(f"{epoch+1:7} {train_loss:12.4f} {train_acc:12.4f} | {val_loss:10.4f} {val_acc:10.4f}")

    # Verifica o melhor modelo até agora
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = resnet18.state_dict()

print("\nTreinamento finalizado.")
print(f"Melhor acurácia de validação: {best_val_acc:.4f}")



# Salva o melhor modelo
torch.save(best_model_wts, modelo_path)
print(" Melhor modelo salvo.")


# Avaliação
resnet18.eval()
val_loss, val_acc = 0, 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)
        val_acc += accuracy(outputs, labels) * inputs.size(0)

val_loss /= val_size
val_acc /= val_size


print(f"{epoch+1:>7} {train_loss:12.4f} {train_acc:12.4f} | {val_loss:10.4f} {val_acc:10.4f}")

# Se a acurácia de validação for melhor, salva o modelo
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(resnet18.state_dict(), modelo_path)
    print(f" melhor modelo salvo {epoch+1} com Val Acc = {val_acc:.4f}")


    # Relatório de Classificação e Matriz de Confusão
resnet18.load_state_dict(torch.load(modelo_path))
resnet18.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n Relatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
print(" Matriz de Confusão:")
print(confusion_matrix(all_labels, all_preds))


# Visualização
def visualizar_amostras(classe, num_imagens=6):
    idxs = [i for i, (_, label) in enumerate(full_dataset) if full_dataset.classes[label] == classe]
    if not idxs:
        print(f"Nenhuma imagem encontrada para a classe: {classe}")
        return

    amostras = random.sample(idxs, min(num_imagens, len(idxs)))

    resnet18.eval()
    with torch.no_grad():
        for idx in amostras:
            img, _ = full_dataset[idx]
            input_img = img.unsqueeze(0).to(device)
            output = resnet18(input_img)
            _, pred = torch.max(output, 1)
            pred_label = full_dataset.classes[pred.item()]

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(img.device)
            img_show = img * std + mean
            img_show = img_show.clamp(0, 1)

            titulo = "Pedestre" if pred_label == "pedestrian" else "Não Pedestre"

            plt.figure(figsize=(4,4))
            plt.title(titulo)
            plt.imshow(np.transpose(img_show.cpu().numpy(), (1, 2, 0)))
            plt.axis('off')
            plt.show()


            
