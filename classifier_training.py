import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from classifier_orig import MNISTClassifier, CIFAR10Classifier
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def evaluate(model_path, dataset='mnist', device='cuda'):
    # Define transform
    transform = transforms.ToTensor()

    # Load dataset and corresponding model
    if dataset == 'mnist':
        test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)
        model = MNISTClassifier()
    elif dataset == 'cifar10':
        test_data = datasets.CIFAR10(root='.', train=False, download=True, transform=transform)
        model = CIFAR10Classifier()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Prepare DataLoader
    loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Load model weights and move to device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Evaluating on {dataset.upper()}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # logits (no softmax)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"{dataset.upper()} Test Accuracy: {accuracy:.2f}%")
    return accuracy

def train(dataset='mnist', epochs=10):
    transform = transforms.ToTensor()
    if dataset == 'mnist':
        train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        model = MNISTClassifier()
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root='.', train=True, download=True, transform=transform)
        model = CIFAR10Classifier()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(train_data, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f"saved_models{dataset}_classifier.pth")
    print(f"Saved {dataset.upper()} classifier")

    # Evaluate the model after training
    evaluate(model, dataset, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.dataset == 'cifar10':
    # Skip training, just evaluate using saved weights
        evaluate(model_path='saved_models/cifar10_classifier.pth', dataset='cifar10', device='cuda')
    else:
    # Train and evaluate MNIST classifier
        train(args.dataset, args.epochs)
