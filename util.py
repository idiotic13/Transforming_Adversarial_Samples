import os
import warnings
import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt
from subprocess import call
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock

from sklearn.model_selection import train_test_split



# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Gaussian noise scale sizes
STDEVS = {
    'mnist': {'FGSM': 0.310, 'bim-a': 0.128, 'bim-b': 0.265, 'pgd': 0.150, 'all': 0.300},
    'cifar10': {'FGSM': 0.050, 'bim-a': 0.009, 'bim-b': 0.039, 'pgd': 0.050, 'all': 0.050},
    'svhn': {'FGSM': 0.132, 'bim-a': 0.015, 'bim-b': 0.122, 'pgd': 0.132, 'all': 0.132}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def conv_layers(self, x):
        out = F.relu(self.bn_conv1(self.conv1(x)))
        out = F.relu(self.bn_conv2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        out = F.relu(self.bn_conv3(self.conv3(out)))
        out = F.relu(self.bn_conv4(self.conv4(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        return out

    def dense_layers(self, x):
        out = F.relu(self.bn_dense1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn_dense2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 256 * 8 * 8)
        out = self.dense_layers(out)
        return out

########################################################################
# 2. Define other models based on dataset
########################################################################

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),  # (N, 1, 28, 28) -> (N, 64, 26, 26)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),            # (N, 64, 26, 26) -> (N, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (N, 64, 24, 24) -> (N, 64, 12, 12)
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

class SVHNModel(nn.Module):
    def __init__(self):
        super(SVHNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        # Compute feature size approximately for 32x32 input (can vary; adjust as needed)
        # After conv1: 30x30; after conv2: 28x28; after pool: 14x14; flatten: 64*14*14
        self.fc1 = nn.Linear(64 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        features = F.relu(self.fc1(x))
        x = self.dropout(features)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if return_features:
            return features
        return x

def get_model(dataset='mnist'):
    assert dataset in ['mnist', 'cifar10'], "Dataset must be 'mnist', 'cifar'"
    if dataset == 'mnist':
        return MNISTClassifier().to(DEVICE)
    elif dataset == 'cifar10':
        return CIFAR10Classifier().to(DEVICE)
    else:  # svhn
        return SVHNModel().to(DEVICE)

########################################################################
# 3. Data loading and preprocessing
########################################################################
def get_data(adv_path):
    data = torch.load(adv_path, map_location='cpu')
    # Expect keys 'adv_train' and 'adv_test', each a dict with 'clean', 'adv', 'labels'
    adv_train = data['adv_train']
    adv_test  = data['adv_test']

    # Extract train arrays
    X_train_clean = adv_train['clean'].numpy()
    X_train_adv   = adv_train['adv'].numpy()
    y_train       = adv_train['labels'].numpy()

    # Extract test arrays
    X_test_clean  = adv_test['clean'].numpy()
    X_test_adv    = adv_test['adv'].numpy()
    y_test        = adv_test['labels'].numpy()

    # Print shapes
    print(f"X_train_clean shape: {X_train_clean.shape}")
    print(f"X_train_adv shape:   {X_train_adv.shape}")
    print(f"y_train shape:       {y_train.shape}")
    print(f"X_test_clean shape:  {X_test_clean.shape}")
    print(f"X_test_adv shape:    {X_test_adv.shape}")
    print(f"y_test shape:        {y_test.shape}")

    return X_train_clean, X_train_adv, y_train, X_test_clean, X_test_adv, y_test

########################################################################
# 4. Utility functions for adversarial/noise operations
########################################################################

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples.
    x: numpy array
    nb_diff: number of pixels to flip
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff, replace=False)
    x[inds] = 1.
    return np.reshape(x, original_shape)

def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    Generate noisy samples based on an adversarial sample or using Gaussian noise.
    """
    if attack in ['jsma', 'cw']:
        X_test_noisy = np.zeros_like(X_test)
        # Use tqdm to show progress through each test sample.
        for i in tqdm(range(len(X_test)), desc="Generating noisy samples (jsma/cw)"):
            # Count the number of pixels that are different.
            nb_diff = np.sum(X_test[i] != X_test_adv[i])
            # Randomly flip an equal number of pixels.
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Using pre-set Gaussian scale sizes to craft noisy samples. "
                      "If you've altered the eps/eps-iter parameters of the attacks used, "
                      "you'll need to update these. In the future, scale sizes will be inferred automatically.")
        # Add Gaussian noise to the samples (clip between 0 and 1)
        noise = np.random.normal(loc=0, scale=STDEVS[dataset][attack], size=X_test.shape)
        X_test_noisy = np.clip(X_test + noise, 0, 1)
    return X_test_noisy

########################################################################
# 5. Monte Carlo Dropout Predictions
########################################################################

# def get_mc_predictions(model, X, n_drop=50, batch_size=256):
#     """
#     Performs Monte Carlo dropout by enabling dropout at inference time.
#     This function temporarily sets the model to train mode so that dropout is active.
    
#     Args:
#         model: PyTorch model with dropout layers.
#         X: numpy array of inputs.
#         n_drop: Number of stochastic forward passes.
#         batch_size: Batch size.
        
#     Returns:
#         Array of shape (n_drop, len(X), num_classes) containing MC predictions.
#     """
#     # enable dropout by setting the model to train mode.
#     model.train()
#     preds_mc = []
#     model.eval_mode_backup = model.training  # backup flag if needed

#     # Create DataLoader from numpy array
#     dataset = torch.tensor(X, dtype=torch.float32)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Outer loop for n_drop passes with a tqdm progress bar.
#     for _ in tqdm(range(n_drop), desc="MC Dropout Passes"):
#         preds = []
#         # Loop through batches (inner loop - no tqdm here to avoid clutter)
#         for batch in loader:
#             batch = batch.to(DEVICE)
#             with torch.no_grad():
#                 # Forward pass with dropout enabled
#                 output = model(batch)
#                 preds.append(output.cpu().numpy())
#         preds_mc.append(np.concatenate(preds, axis=0))
#     # Optionally set back to eval mode.
#     model.eval()
#     return np.array(preds_mc)

# ########################################################################
# # 6. Deep Representations Extraction using Forward Hooks
# ########################################################################

# def get_deep_representations(model, X, batch_size=256):
#     """
#     Extracts the deep representations from the penultimate layer of the model.
#     In our PyTorch models, we assume that if you call model(x, return_features=True),
#     it will return the desired internal features.
#     """
#     model.eval()
#     representations = []
#     dataset = torch.tensor(X, dtype=torch.float32)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     # Use tqdm to show batch processing for representations extraction.
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Extracting deep representations"):
#             batch = batch.to(DEVICE)
#             # Assumes that model(x, return_features=True) returns features.
#             feats = model(batch, return_features=True)
#             representations.append(feats.cpu().numpy())
#     representations = np.concatenate(representations, axis=0)
#     return representations

########################################################################
# 7. Scoring functions using multiprocessing
########################################################################

def score_point(tup):
    """
    Helper function: returns the log density score for a single sample.
    """
    x, kde = tup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]

def score_samples(kdes, samples, preds, n_jobs=None):
    """
    Compute scores for each sample using the provided KDEs with multiprocessing.
    """
    import multiprocessing as mp
    pool = mp.Pool(n_jobs) if n_jobs is not None else mp.Pool()
    results = np.asarray(pool.map(
        score_point,
        [(x, kdes[i]) for x, i in zip(samples, preds)]
    ))
    pool.close()
    pool.join()
    return results

########################################################################
# 8. Normalize representations (scaling)
########################################################################

def normalize(normal, adv, noisy):
    """
    Normalize three sets of samples together using sklearn's scale function.
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))
    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

########################################################################
# 9. Train a Logistic Regression classifier using scikit-learn
########################################################################

# def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
#     """
#     Train a logistic regression classifier on the density and uncertainty values.
#     """
#     values_neg = np.concatenate(
#         (densities_neg.reshape((1, -1)),
#          uncerts_neg.reshape((1, -1))),
#         axis=0).transpose()
#     values_pos = np.concatenate(
#         (densities_pos.reshape((1, -1)),
#          uncerts_pos.reshape((1, -1))),
#         axis=0).transpose()
    
#     values = np.concatenate((values_neg, values_pos))
#     labels = np.concatenate((np.zeros_like(densities_neg), np.ones_like(densities_pos)))
    
#     lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)
#     return values, labels, lr

def train_lr(
    densities_pos, densities_neg,
    uncerts_pos,    uncerts_neg,
    test_size=0.3,
    random_state=42
):
    """
    Split your detection‐features into train/val, fit a logistic detector,
    and return both splits plus the trained model.
    """
    # 1) Stack your two features (dual‐manifold score, MC‐uncertainty)
    vals_neg = np.stack((densities_neg, uncerts_neg), axis=1)
    vals_pos = np.stack((densities_pos, uncerts_pos), axis=1)

    X = np.concatenate([vals_neg, vals_pos], axis=0)
    y = np.concatenate([
        np.zeros(len(densities_neg), dtype=int),
        np.ones(len(densities_pos),  dtype=int)
    ], axis=0)

    # 2) Split into train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 3) Fit logistic regression (with internal CV)
    lr = LogisticRegressionCV(
        cv=5,        # you can adjust number of folds
        n_jobs=-1,
        random_state=random_state
    ).fit(X_train, y_train)

    return X_train, y_train, X_val, y_val, lr

########################################################################
# 10. Compute ROC curve and AUC
########################################################################

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    Compute and optionally plot the ROC curve and AUC score.
    """
    p_scores = np.concatenate((probs_neg, probs_pos))
    y_true = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, ths = roc_curve(y_true, p_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, p_scores)

    prec = precision[1:]   # drop precision[0], which has no threshold
    rec  = recall[1:]      # drop recall[0]
    f1   = 2 * (prec * rec) / (prec + rec + 1e-16)  # add small eps to avoid 0/0

    # Pick the threshold with the highest F1
    best_idx       = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    best_f1        = f1[best_idx]
    best_prec      = prec[best_idx]
    best_rec       = rec[best_idx]

    print(f"Best F1 = {best_f1:.3f}")
    print(f"  at threshold = {best_threshold:.3f}")
    print(f"  precision = {best_prec:.3f}, recall = {best_rec:.3f}")

    # accs = [(probs >= t).mean() == labels.mean()  for t in ths]  # faster with numpy
    # # more explicitly:
    # accs = [accuracy_score(labels, probs >= t) for t in ths]

    # best_idx   = np.argmax(accs)
    # best_thr   = ths[best_idx]
    # best_acc   = accs[best_idx]

    # print(f"Best threshold={best_thr:.3f} → accuracy={best_acc:.2%}")

    auc_score = auc(fpr, tpr)
    # if plot:
    #     plt.figure(figsize=(7, 6))
    #     plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
    #     plt.legend(loc='lower right')
    #     plt.title("ROC Curve")
    #     plt.xlabel("FPR")
    #     plt.ylabel("TPR")
    #     plt.show()
    return fpr, tpr, auc_score

def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# def get_dual_manifold_scores(X_features, preds, manifolds_clean, manifolds_adv):
#     scores = []
#     for x, label in zip(X_features, preds):
#         clean_m = manifolds_clean[label]['mean']
#         clean_cov_inv = np.linalg.pinv(manifolds_clean[label]['cov'])

#         adv_m = manifolds_adv[label]['mean']
#         adv_cov_inv = np.linalg.pinv(manifolds_adv[label]['cov'])

#         d_clean = mahalanobis_distance(x, clean_m, clean_cov_inv)
#         d_adv = mahalanobis_distance(x, adv_m, adv_cov_inv)

#         scores.append(d_clean - d_adv)
#     return np.array(scores)

def get_dual_manifold_scores(
    X_features, preds,
    manifolds_clean, manifolds_adv,
    batch_size=1024,
    device='cuda'
):
    """
    Compute dual-manifold scores in batches on GPU (if available).
    X_features: NumPy array of shape (N, D)
    preds:       NumPy array of shape (N,) with integer class labels
    manifolds_clean/adv: dict mapping class -> {'mean': np.array(D,), 'cov': np.array(D,D)}

    Returns a NumPy array of shape (N,) with score = d_clean - d_adv.
    """
    # Move data to torch
    X = torch.from_numpy(X_features).float().to(device)
    P = torch.from_numpy(preds).long().to(device)

    # Prepare class-wise tensors
    num_classes = len(manifolds_clean)
    # Stack means and inverse covariances
    means_clean = torch.stack([
        torch.from_numpy(manifolds_clean[c]['mean']).float()
        for c in range(num_classes)
    ]).to(device)                           # [C, D]
    covinv_clean = torch.stack([
        torch.from_numpy(np.linalg.pinv(manifolds_clean[c]['cov']).astype(np.float32))
        for c in range(num_classes)
    ]).to(device)                           # [C, D, D]
    means_adv = torch.stack([
        torch.from_numpy(manifolds_adv[c]['mean']).float()
        for c in range(num_classes)
    ]).to(device)
    covinv_adv = torch.stack([
        torch.from_numpy(np.linalg.pinv(manifolds_adv[c]['cov']).astype(np.float32))
        for c in range(num_classes)
    ]).to(device)

    scores_batches = []
    N = X.shape[0]
    for start in tqdm(range(0, N, batch_size), desc="Dual manifold scoring"):
        end = min(start + batch_size, N)
        Xb = X[start:end]                       # [B, D]
        Pb = P[start:end]                       # [B]

        # Gather class-specific mean & cov-inv
        mc = means_clean[Pb]                    # [B, D]
        ma = means_adv[Pb]                      # [B, D]
        ccinv_b = covinv_clean[Pb]             # [B, D, D]
        advinv_b = covinv_adv[Pb]              # [B, D, D]

        # Compute Mahalanobis distances: sqrt((x-m)T @ covinv @ (x-m))
        dclean2 = torch.einsum('bi,bij,bj->b', Xb - mc, ccinv_b, Xb - mc)
        dadv2   = torch.einsum('bi,bij,bj->b', Xb - ma, advinv_b, Xb - ma)
        dclean  = torch.sqrt(dclean2)
        dadv    = torch.sqrt(dadv2)

        scores_batches.append((dclean - dadv).cpu())

    scores = torch.cat(scores_batches).numpy()
    return scores


def evaluate_model(model, dataset, labels, name, batch_size, device):
    model.eval()
    dataloader = DataLoader(
        TensorDataset(torch.tensor(dataset).float(), torch.tensor(labels)),
        batch_size=batch_size
    )
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.argmax(dim=1))
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = (all_preds == all_targets).float().mean().item()
    print(f"Model accuracy on the {name} test set: {100 * acc:.2f}%")

def extract_features(model, x, dataset):
    """
    Generic feature extractor for MNISTClassifier and CIFAR10Classifier.
    Returns the activations from the penultimate layer.
    """
    model.eval()
    with torch.no_grad():
        if dataset == 'mnist':
            # MNISTClassifier.model is nn.Sequential with layers:
            # [Conv2d, ReLU, Conv2d, ReLU, MaxPool2d, Dropout, Flatten,
            #  Linear(64*12*12->128), ReLU, Dropout, Linear(128->10)]
            # We take all layers up to the final Dropout (exclude last Linear)
            # to get the 128-dimensional feature.
            features = model.model[:10](x)

        elif dataset == 'cifar10':
            # CIFAR10Classifier: conv_layers -> flatten -> fc1 -> bn -> ReLU -> dropout -> fc2 -> bn -> ReLU -> dropout -> fc3
            # We take activations after the second dense ReLU (exclude final fc3)
            out = model.conv_layers(x)
            out = out.view(out.size(0), -1)
            out = F.relu(model.bn_dense1(model.fc1(out)))
            out = model.dropout(out)
            features = F.relu(model.bn_dense2(model.fc2(out)))

        else:
            raise ValueError(f"Unsupported dataset for feature extraction: {dataset}")

    return features

def get_deep_features(model, X, dataset, batch_size=32, save_file=None, device='cpu'):
    """
    Extract deep features in batches.  
    If save_file exists, loads and returns it.  
    Otherwise computes features via extract_features, saves to .npy if save_file given.
    Returns a NumPy array of shape (N, feature_dim).
    """
    if save_file is not None and os.path.isfile(save_file):
        print(f"Loading deep features from {save_file}")
        return np.load(save_file)

    model.to(device)
    model.eval()
    feature_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).float().to(device)
            feats = extract_features(model, batch, dataset)
            feature_list.append(feats.cpu().numpy())
            del batch, feats
            torch.cuda.empty_cache()

    features = np.vstack(feature_list)

    if save_file is not None:
        np.save(save_file, features)
        print(f"Saved deep features to {save_file}")

    return features

def get_mc_uncertainties(model, X, device, batch_size=32, mc_runs=50, save_file=None):
    """
    Compute MC dropout uncertainties. If `save_file` exists, load from it.
    Otherwise compute and save.
    """
    if save_file is not None and os.path.isfile(save_file):
        print(f"Loading MC uncertainties from {save_file}")
        return np.load(save_file)

    # Enable dropout, but keep all BatchNorm layers in eval(since we only need to change dropout)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # model.train()  # Enable dropout
    preds_list = []
    with torch.no_grad():
        for _ in range(mc_runs):
            batch_preds = []
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size]).float().to(device)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                batch_preds.append(probs.cpu().numpy())
                del batch, logits, probs
                torch.cuda.empty_cache()
            preds_list.append(np.concatenate(batch_preds, axis=0))
    preds_array = np.array(preds_list)  # shape: (mc_runs, n_samples, n_classes)
    uncert = preds_array.var(axis=0).mean(axis=1)

    model.eval()
    if save_file is not None:
        np.save(save_file, uncert)
        print(f"Saved MC uncertainties to {save_file}")
    return uncert
