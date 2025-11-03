import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=0)  
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)           
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                

    
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=0)         
        self.tanh3 = nn.Tanh()

        self.fc1 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.tanh1(x)
        x=self.pool1(x)
        
        x=self.conv2(x)
        x=self.tanh2(x)
        x=self.pool2(x)

        x=self.conv3(x)
        x=self.tanh3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh4(x)
        x = self.fc2(x)
        return x


# -------------------------
#train, validate, save, plot
# -------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc="Train", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Valid", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(state, filename):
    torch.save(state, filename)

def plot_metrics(history, out_dir):
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics.png'))
    plt.close()
   
# -------------------------
# Main entrypoint
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets and loaders
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    val_set   = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = LeNet5(num_classes=10, in_channels=3)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.output_dir, 'best_lenet_cifar10.pth'))

        # also save last
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, os.path.join(args.output_dir, 'last_lenet_cifar10.pth'))

        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f" Best Val Acc so far: {best_acc:.2f}%\n")

    # plot metrics
    plot_metrics(history, args.output_dir)
    print("Training finished. Best val acc: {:.2f}%".format(best_acc))
    print("Checkpoints and metrics saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeNet-5 on CIFAR-10 (PyTorch)")
    parser.add_argument('--data-dir', type=str, default='./data', help='directory to download CIFAR-10')
    parser.add_argument('--output-dir', type=str, default=f'./results/output_lenet_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='where to save models/plots')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--lr-step', type=int, default=15)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA even if available')
    args = parser.parse_args()
    main(args)