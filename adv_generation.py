import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torchattacks

# Import your model definitions from a separate file
from classifier import MNISTClassifier, CIFAR10Classifier

# Basic Iterative Method implementation for BIM-A and BIM-B
def basic_iterative_method(model, X, Y, eps, eps_iter, nb_iter=10, batch_size=256, clip_min=0.0, clip_max=1.0, device='cpu'):
    """Basic Iterative Method (BIM) in PyTorch."""
    model.eval()

    # Convert inputs to tensors if they are numpy arrays
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float().to(device)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).long() if Y.ndim == 1 else torch.from_numpy(Y).float()
        Y = Y.to(device)

    X_adv = X.clone().detach()
    results = torch.zeros((nb_iter,) + X.shape, device=device)
    its = {}
    out = set()

    for i in tqdm(range(nb_iter), desc="BIM Progress"):
        X_adv.requires_grad_(True)

        outputs = model(X_adv)
        loss = F.cross_entropy(outputs, Y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = X_adv.grad.data
            X_adv = X_adv + eps_iter * grad.sign()

            # Clip to maintain the perturbation within epsilon
            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
            X_adv = torch.clamp(X + delta, min=clip_min, max=clip_max).detach()

            results[i] = X_adv.clone()

            # Track misclassifications
            preds = model(X_adv)
            predicted_labels = torch.argmax(preds, dim=1)
            true_labels = Y if Y.ndim == 1 else torch.argmax(Y, dim=1)

            misclassified = (predicted_labels != true_labels).nonzero(as_tuple=True)[0]

            for idx in misclassified.tolist():
                if idx not in out:
                    its[idx] = i
                    out.add(idx)

    return its, results.cpu().numpy()

# Define attack parameters per dataset (no CW)
ATTACK_PARAMS = {
    'mnist': {
        'FGSM':  {'eps': 0.3},
        'PGD':   {'eps': 0.3,   'alpha': 0.01,  'steps': 40},
        'BIM-A': {'eps': 0.3,   'alpha': 0.01,  'steps': 10},
        'BIM-B': {'eps': 0.3,   'alpha': 0.01,  'steps': 10},
    },
    'cifar10': {
        'FGSM':  {'eps': 0.03},
        'PGD':   {'eps': 0.03,  'alpha': 2/255, 'steps': 40},
        'BIM-A': {'eps': 0.03,  'alpha': 2/255, 'steps': 10},
        'BIM-B': {'eps': 0.03,  'alpha': 2/255, 'steps': 10},
    }
}


def make_transform(dataset):
    # No normalization—just convert to tensor
    return transforms.ToTensor()


def generate_adv_samples(model, loader, attack, params, device):
    model.eval()
    clean, adv, labels = [], [], []

    for x, y in tqdm(loader, desc=f"Generating {attack}"):
        x, y = x.to(device), y.to(device)

        if attack in ['FGSM', 'PGD']:
            # use torchattacks for FGSM and PGD
            atk = getattr(torchattacks, attack)(model, **params)
            x_adv = atk(x, y)

        elif attack in ['BIM-A', 'BIM-B']:
            # custom BIM-A / BIM-B
            eps, eps_iter = params['eps'], params['alpha']
            steps = params['steps']
            its, results = basic_iterative_method(
                model, x.clone().detach(), y,
                eps=eps, eps_iter=eps_iter,
                nb_iter=steps, batch_size=x.size(0), device=device
            )
            if attack == 'BIM-A':
                # take first misclassification
                arr = np.stack([results[its[i], i] if i in its else results[-1, i]
                                for i in range(len(y))])
            else:
                # BIM-B: always last step
                arr = results[-1]
            x_adv = torch.from_numpy(arr).float().to(device)

        else:
            raise ValueError(f"Unsupported attack: {attack}")

        clean.append(x.cpu())
        adv.append(x_adv.cpu())
        labels.append(y.cpu())

    return {
        'clean': torch.cat(clean, dim=0),
        'adv':   torch.cat(adv,   dim=0),
        'labels':torch.cat(labels, dim=0)
    }


def generate(dataset, attack, model_path, batch_size=256):
    transform = make_transform(dataset)

    if dataset == 'mnist':
        train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        test_data  = datasets.MNIST(root='.', train=False, download=True, transform=transform)
        model = MNISTClassifier()
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root='.', train=True, download=True, transform=transform)
        test_data  = datasets.CIFAR10(root='.', train=False, download=True, transform=transform)
        model = CIFAR10Classifier()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    params = ATTACK_PARAMS[dataset][attack]
    adv_train = generate_adv_samples(model, train_loader, attack, params, device)
    adv_test  = generate_adv_samples(model, test_loader,  attack, params, device)

    output_file = f"adv_data/{dataset}_{attack}_adv.pt"
    torch.save({'adv_train': adv_train, 'adv_test': adv_test}, output_file)
    print(f"Saved adversarial data to {output_file}")


if _name_ == '_main_':
    parser = argparse.ArgumentParser(description='Generate adversarial examples')
    parser.add_argument('--dataset',   choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--attack',    choices=['FGSM', 'PGD', 'bim-a', 'bim-b'], required=True)
    parser.add_argument('--model_path', help='Path to saved model .pth file', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    generate(args.dataset, args.attack, args.model_path, args.batch_size)