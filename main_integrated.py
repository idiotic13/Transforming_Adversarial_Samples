import os
import torch
import argparse
from torch.utils.data import random_split, DataLoader
from torch.nn import DataParallel
import torchvision.transforms as transforms

from model import get_model
from train_integrated import train, val, test
from data_loader import AdversarialTensorDataset
from inferencer import Inferencer


def main():
    parser = argparse.ArgumentParser(description="Adversarial training or testing")
    parser.add_argument('--phase',   choices=['train', 'test'], required=True)
    parser.add_argument('--attack',      required=True, help='Attack name (fgsm, pgd, etc.)')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--feat_layer', required=True,
                        help='Feature layer name for guided loss (e.g. pre_fc2)')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Total number of epochs for training')

    # for inferencer
    parser = argparse.ArgumentParser(description="Run adversarial‐detector inference")
    parser.add_argument('--adv_path',    required=True, help='Path to adv .pt file (torch.save dict)')
    parser.add_argument('--model_path',  required=True, help='Path to classifier .pth')
    parser.add_argument('--save_dir',    required=True, help='Dir containing lr_model and manifolds')
    parser.add_argument('--mc_runs',     type=int, default=50, help='MC‐dropout runs')
    parser.add_argument('--threshold',   type=float, default=0.25, help='Decision threshold')
    parser.add_argument('--num_samples', type=int, default=1000, help='# of clean/adv to sample each')
    parser.add_argument('--batch_size',  type=int, default=64, help='Batch size for predict_batch')
    parser.add_argument('--seed',        type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    phase        = args.phase
    attack       = args.attack
    dataset      = args.dataset
    feat_layer   = args.feat_layer
    total_epochs = args.epochs

    # Log file
    log_file = f"{dataset}_{attack}_{phase}.out"
    os.makedirs("checkpoints", exist_ok=True)

    # ─── Model Setup ──────────────────────────────────────
    config, net = get_model(dataset=dataset, feat_layer=feat_layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DataParallel(net).to(device)

    if phase == 'test':
        model_path = f"checkpoints/{dataset}_{attack}_best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model for testing: {model_path}")
    else:
        checkpoint_path = f"checkpoints/{dataset}_{attack}_latest_model.pth"
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Resuming training from: {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"[WARNING] No checkpoint found at {checkpoint_path}. Starting from scratch.")

    # ─── Load Dataset ──────────────────────────────────────
    transform = None  # data already in tensor form
    full_data = AdversarialTensorDataset(dataset, attack, phase=('train' if phase=='train' else 'test'),
                                         transform=transform)

    if phase == 'train':
        train_size = int(0.8 * len(full_data))
        val_size   = len(full_data) - train_size
        train_ds, val_ds = random_split(full_data, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                                  num_workers=4, pin_memory=True)
    else:
        test_loader = DataLoader(full_data, batch_size=64, shuffle=False,
                                 num_workers=4, pin_memory=True)

    # ─── Optimizer ────────────────────────────────────────
    optimizer = torch.optim.Adam(net.module.denoise.parameters(), lr=1e-3)

    def get_lr(epoch):
        return 1e-4 if epoch <= 0.9 * total_epochs else 1e-5

    # ─── Training or Testing ───────────────────────────────
    if phase == 'train':
        best_val_acc = 0.0
        for epoch in range(1, total_epochs + 1):
            train_orig_acc, train_adv_acc, train_losses = train(
                epoch, net, train_loader, optimizer, get_lr
            )
            val_orig_acc, val_adv_acc, val_losses = val(
                epoch, net, val_loader
            )

            # Save latest checkpoint
            latest_ckpt = f"checkpoints/{dataset}_{attack}_latest_model.pth"
            torch.save(net.state_dict(), latest_ckpt)

            # Save best model
            if val_adv_acc > best_val_acc:
                best_val_acc = val_adv_acc
                best_ckpt = f"checkpoints/{dataset}_{attack}_best_model.pth"
                torch.save(net.state_dict(), best_ckpt)

            # Logging
            if epoch % 5 == 0 or epoch == total_epochs:
                with open(log_file, 'a') as f:
                    f.write(
                        f"Epoch {epoch}: "
                        f"Train Orig={train_orig_acc:.4f}, Train Adv={train_adv_acc:.4f} | "
                        f"Val Orig={val_orig_acc:.4f}, Val Adv={val_adv_acc:.4f}\n"
                    )

    else:  # test
        detector = Inferencer(
            dataset    = args.dataset,
            model_path = args.model_path,
            save_dir   = args.save_dir,
            attack     = args.attack,
            mc_runs    = args.mc_runs,
            threshold  = args.threshold
        )
        
        result_path = f"checkpoints/final_test_{dataset}_{attack}.npz" 
        clean_acc, adv_acc, denoised_acc= test(net, detector, test_loader, result_path, args.batch_size)
        # clean_acc, adv_acc, denoised_acc= test(net, test_loader, result_path)
        with open(log_file, 'w') as f:
            f.write(f"Test Clean Accuracy: {clean_acc:.4f}\n")
            f.write(f"Test Adversaril Accuracy: {adv_acc:.4f}\n")
            f.write(f"Test Denoised Accuracy: {denoised_acc:.4f}\n")


if __name__ == '__main__':
    main()
