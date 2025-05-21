import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

class sANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.dropout1(self.fc1(x)))
        x = torch.relu(self.dropout2(self.fc2(x)))
        return self.fc3(x)

model   = sANN().to(device)
scaler  = GradScaler()                        
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def quantization(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    levels = 2 ** n_bits
    return torch.round(x * (levels - 1)) / (levels - 1)


#def bucket_quantize_tensor(x: torch.Tensor, bucket_size: int, n_bits: int) -> torch.Tensor:
#    flat = x.view(-1)
#    q    = flat.clone()
#    levels = 2 ** n_bits
#    for i in range(0, flat.numel(), bucket_size):
#        seg = flat[i : i + bucket_size]
#        q[i : i + bucket_size] = torch.round(seg * (levels - 1)) / (levels - 1)
#    return q.view_as(x)

def evaluate(model, testloader, criterion, device):
    model.eval()

    torch.cuda.reset_peak_memory_stats()

    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            correct += (preds == label).sum().item()
            total += label.size(0)


    acc = correct / total


    return acc

def ann(trainloader, testloader, model, criterion, optimizer, scaler, epochs=30):
    Loss_train, Acc_train = [], []

    for epoch in tqdm(range(epochs), desc="entrainement..."):
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            all_preds.extend(outputs.argmax(1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        Loss_train.append(running_loss / len(trainloader))
        Acc_train.append(accuracy_score(all_labels, all_preds))

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Loss_train, label="Train Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(Acc_train, label="Train Acc")
    plt.legend()
    plt.show()

    return model


model = ann(trainloader, testloader, model, criterion, optimizer, scaler, epochs=30)

test_loss, test_acc = evaluate(model, testloader, criterion, device)
print(f"before quantization: Acc: {test_acc*100:.2f}%")

with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        w = param.data               # tenseur de poids
        a = w.abs()                  # magnitude

        # Seuils pour 3 buckets
        thr_low  = a.quantile(0.33)  
        thr_high = a.quantile(0.66)  

        # Masques
        mask_small  = (a <= thr_low)
        mask_medium = (a > thr_low) & (a <= thr_high)
        mask_large  = (a > thr_high)

        # Copie de travail
        w_q = w.clone()

        # Quantisation des petits et moyens poids
        w_q[mask_small ] = quantization(w[mask_small ], 2)  # 2 bits
        w_q[mask_medium] = quantization(w[mask_medium], 4)  # 4 bits
        # Les gros poids restent inchangés (FP32)

        # Remplacement
        param.data = w_q


test_loss, test_acc = evaluate(model, testloader, criterion, device)
print(f"after quantization: Acc: {test_acc*100:.2f}%")