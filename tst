# fast_train_weight_only_bucketed.py

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms and loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, pin_memory=True)

# MLP model definition
class sANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn1(self.dropout1(self.fc1(x))))
        x = torch.relu(self.bn2(self.dropout2(self.fc2(x))))
        return self.fc3(x)

def bucketed_weights_forward(model: nn.Module, bits_per_bucket: int):
    """
    Vectorized bucketed mantissa reduction for *weights only* (skips biases).
    """
    for name, param in model.named_parameters():
        # Only quantize weight tensors (ndim >= 2), skip biases/1D params
        if not param.requires_grad or param.ndim < 2:
            continue

        W = param.data
        W_abs = W.abs()
        w_max = W_abs.max()
        if w_max == 0:
            continue

        # Number of buckets
        B = math.ceil(23.0 / bits_per_bucket)
        step = w_max / B

        # Compute bucket index for each weight element
        bucket_idx = ((W_abs / step).floor()).clamp(0, B).long()

        # Bits to keep per element (0..23)
        bits = (bucket_idx * bits_per_bucket).clamp(max=23)

        # Decompose into mantissa and exponent
        mant, exp = torch.frexp(W)

        # Quantize mantissa
        two_pow = (2.0 ** bits.to(W.device)).to(W.dtype)
        quant_mant = torch.sign(mant) * torch.floor(torch.abs(mant) * two_pow) / two_pow

        # Reconstruct float32 value
        newW = torch.ldexp(quant_mant, exp)
        param.data.copy_(newW)

def train_with_buckets(model, epochs=15, lr=1e-3, weight_decay=1e-4,
                       step_size=5, gamma=0.5, bits_per_bucket=5):
    """
    Train model with weight-only bucketed mantissa reduction after each batch.
    """
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Quantize only weights after each batch
            bucketed_weights_forward(model, bits_per_bucket)
            loop.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
    return model

def test_inference(model, dtype=torch.float32, runs=100):
    """
    Measure inference time, peak memory, and accuracy for FP32 or FP16.
    """
    model.to(device).eval()
    if dtype == torch.float16:
        model.half()

    images, _ = next(iter(test_loader))
    images = images.to(device)
    if dtype == torch.float16:
        images = images.half()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(images)
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(images)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / runs
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    # Accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            if dtype == torch.float16:
                imgs = imgs.half()
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    accuracy = 100.0 * correct / total

    print(f"=== Inference {dtype} ===")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Avg time  : {avg_time*1e3:.2f} ms")
    print(f"Peak mem  : {peak_mem:.1f} MB\n")

    return avg_time, peak_mem, accuracy

if __name__ == "__main__":
    print(f"Device: {device}\n")

    model = sANN()
    print("Training with weight-only bucketed approximation...")
    model = train_with_buckets(model, epochs=15, bits_per_bucket=5)

    torch.save(model.state_dict(), "model_weight_bucketed.pth")

    print("\nTesting:")
    test_inference(model)
