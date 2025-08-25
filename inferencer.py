import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from util import get_model
from util import *
import argparse

class Inferencer:
    """
    Loads a trained classifier, logistic-regression detector, and manifolds to predict
    whether a given sample is adversarial.
    """
    def __init__(
        self,
        dataset: str,
        model_path: str,
        save_dir: str,
        attack: str,
        mc_runs: int = 50,
        device: torch.device = None,
        threshold: float = 0.15
    ):
        self.dataset = dataset
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load classifier
        self.model = get_model(dataset)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load logistic regression detector
        lr_path = os.path.join(save_dir, f'lr_model_{dataset}_{attack}.pkl')
        with open(lr_path, 'rb') as f:
            self.detector = pickle.load(f)

        # Load manifolds
        manifolds_path = os.path.join(save_dir, f'manifolds_{dataset}_{attack}.pkl')
        with open(manifolds_path, 'rb') as f:
            manifolds = pickle.load(f)
        self.manifolds_clean = manifolds['clean']
        self.manifolds_adv   = manifolds['adv']

        self.mc_runs = mc_runs
        self.threshold = threshold

    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Batch‐wise adversarial detection using GPU acceleration.

        Args:
            X: np.ndarray of shape (N, C, H, W)
            batch_size: how many samples to process at once
            return_proba: if True, return adversarial probabilities;
                        otherwise return boolean flags.

        Returns:
            np.ndarray of shape (N,) containing either bools or floats.
        """
        N = X.shape[0]
        results = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb_np = X[start:end]  # shape (B, C, H, W)
            Xb = torch.from_numpy(Xb_np).float().to(self.device)

            # 1) Classification
            with torch.no_grad():
                logits = self.model(Xb)
                labels = logits.argmax(dim=1)  # (B,)

            # 2) Deep features
            feats = extract_features(self.model, Xb, self.dataset)  # (B, D)

            # 3) MC‐dropout uncertainty (returns np.ndarray)
            uncerts = get_mc_uncertainties(
                self.model,
                Xb_np,  # still pass numpy here as get_mc_uncertainties likely expects it
                self.device,
                batch_size=len(Xb_np),
                mc_runs=self.mc_runs,
                save_file=None
            )  # shape: (B,)

            # 4) Dual‐manifold scoring
            feats_np = feats.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for feat_vec, lbl, unc in zip(feats_np, labels_np, uncerts):
                m_clean = self.manifolds_clean[lbl]['mean']
                cov_clean = self.manifolds_clean[lbl]['cov']
                m_adv   = self.manifolds_adv[lbl]['mean']
                cov_adv = self.manifolds_adv[lbl]['cov']

                d_clean = mahalanobis_distance(feat_vec, m_clean, np.linalg.pinv(cov_clean))
                d_adv   = mahalanobis_distance(feat_vec, m_adv,   np.linalg.pinv(cov_adv))
                dual_score = d_clean - d_adv

                det_input = np.array([[dual_score, float(unc)]])
                adv_proba = float(self.detector.predict_proba(det_input)[0, 1])

                if return_proba:
                    results.append(adv_proba)
                else:
                    results.append(adv_proba > self.threshold)

        return np.array(results)


# def main():
#     parser = argparse.ArgumentParser(description="Run adversarial‐detector inference")
#     parser.add_argument('--adv_path',    required=True, help='Path to adv .pt file (torch.save dict)')
#     parser.add_argument('--dataset',     required=True, choices=['mnist','cifar'], help='Dataset name')
#     parser.add_argument('--model_path',  required=True, help='Path to classifier .pth')
#     parser.add_argument('--save_dir',    required=True, help='Dir containing lr_model and manifolds')
#     parser.add_argument('--attack',      required=True, help='Attack name (fgsm, pgd, etc.)')
#     parser.add_argument('--mc_runs',     type=int, default=50, help='MC‐dropout runs')
#     parser.add_argument('--threshold',   type=float, default=0.25, help='Decision threshold')
#     parser.add_argument('--num_samples', type=int, default=1000, help='# of clean/adv to sample each')
#     parser.add_argument('--batch_size',  type=int, default=64, help='Batch size for predict_batch')
#     parser.add_argument('--seed',        type=int, default=42, help='Random seed for sampling')

#     args = parser.parse_args()

#     # 1) load the adv data
#     X_train_clean, X_train_adv, y_train, X_test_clean, X_test_adv, y_test = get_data(args.adv_path)

#     # 2) sample evenly
#     rng = np.random.default_rng(args.seed)
#     idx_c = rng.choice(len(X_test_clean),  args.num_samples, replace=False)
#     idx_a = rng.choice(len(X_test_adv),    args.num_samples, replace=False)

#     X_samples = np.concatenate([X_test_clean[idx_c], X_test_adv[idx_a]], axis=0)
#     y_labels  = np.concatenate([np.zeros(args.num_samples, dtype=int),
#                                 np.ones(args.num_samples,  dtype=int)], axis=0)

#     # 3) init inferencer
#     infer = Inferencer(
#         dataset    = args.dataset,
#         model_path = args.model_path,
#         save_dir   = args.save_dir,
#         attack     = args.attack,
#         mc_runs    = args.mc_runs,
#         threshold  = args.threshold
#     )

#     # 4) batch‐predict
#     y_pred = infer.predict_batch(X_samples, batch_size=args.batch_size)

#     acc = (y_pred == y_labels).mean() * 100
#     print(f"Detector accuracy on {args.num_samples} clean + {args.num_samples} adv: {acc:.2f}%")

# if __name__ == '__main__':
#     main()
