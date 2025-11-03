import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet import resnet20 

# --- CIFAR10 Setup: Data Augmentation and Normalization ---
# Standard data augmentation for CIFAR-10 training
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),        
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    # Normalization using CIFAR-10 mean and std
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
])

# Standard test set transformation (no augmentation)
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
])

# --- Main Execution Block (Fixes Multiprocessing Error on Windows) ---
if __name__ == '__main__':
    
    # --- Data Loading ---
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # --- Model, Loss, and Optimizer Initialization ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = resnet20().to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Standard SGD optimizer with momentum and weight decay for ResNet training
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # Learning Rate Scheduler: Drops LR by 10x at milestones [80, 120] (Standard for 160 epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    # --- Training Loop ---
    EPOCHS = 164
    best_acc = 0
    
    print(f"Starting training on {device} for {EPOCHS} epochs.")

    for epoch in range(EPOCHS):
        net.train()
        total_loss, correct, total = 0, 0, 0
        
        # Training phase
        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Stepping the scheduler after each epoch
        scheduler.step()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(trainloader)
        print(f"Train acc: {train_acc:.2f}%, loss: {avg_loss:.3f}")

        # --- Evaluation Phase ---
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = net(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Test accuracy: {acc:.2f}% (LR: {scheduler.get_last_lr()[0]:.1e})")

        # Saveing the model if test accuracy improves
        if acc > best_acc:
            best_acc = acc
            print(f"Saving model with new best accuracy: {best_acc:.2f}%")
            torch.save(net.state_dict(), 'best_resnet20_cifar10.pth')

    print(f"\nFinal Best Test Accuracy: {best_acc:.2f}%")