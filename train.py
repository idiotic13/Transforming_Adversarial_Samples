import os
import time
import numpy as np
import torch
from torch.nn import DataParallel


def train(epoch, net, data_loader, optimizer, get_lr):
    """
    Run one epoch of training.

    Returns:
        orig_acc (float): accuracy on clean inputs
        adv_acc  (float): accuracy on denoised adversarial inputs
        avg_loss (float): average feature loss over the epoch
    """
    start_time = time.time()

    # 1) Eval mode for full model, train mode for denoiser only
    net.eval()
    if isinstance(net, DataParallel):
        net.module.denoise.train()
    else:
        net.denoise.train()

    # 2) Adjust learning rate
    lr = get_lr(epoch)
    for g in optimizer.param_groups:
        g['lr'] = lr

    orig_accs, adv_accs, losses = [], [], []

    # 3) Batch loop
    for orig, adv, label in data_loader:
        orig  = orig.cuda(non_blocking=True)
        adv   = adv.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # 4) Forward pass: returns (orig_logits, adv_logits, feat_loss)
        orig_pred, adv_pred, feat_loss = net(orig, adv)

        # 5) Compute accuracies
        orig_accs.append((orig_pred.argmax(1) == label).float().mean().item())
        adv_accs.append((adv_pred.argmax(1) == label).float().mean().item())

        # 6) Backprop on feature loss
        optimizer.zero_grad()
        # ensure scalar
        if feat_loss.dim() != 0:
            loss = feat_loss.mean()
        else:
            loss = feat_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # 7) Aggregate
    orig_acc = float(np.mean(orig_accs))
    adv_acc  = float(np.mean(adv_accs))
    avg_loss = float(np.mean(losses))
    elapsed  = time.time() - start_time

    print(f"Epoch {epoch:3d} (lr {lr:.5f}) — "
          f"orig_acc {orig_acc:.3f}, adv_acc {adv_acc:.3f}, "
          f"loss {avg_loss:.5f}, time {elapsed:.1f}s")

    return orig_acc, adv_acc, avg_loss


def val(epoch, net, data_loader):
    """
    Run one epoch of validation.

    Returns:
        orig_acc (float): accuracy on clean inputs
        adv_acc  (float): accuracy on denoised adversarial inputs
        avg_loss (float): average feature loss over the epoch
    """
    net.eval()
    if isinstance(net, DataParallel):
        net.module.denoise.eval()
    else:
        net.denoise.eval()

    orig_accs, adv_accs, losses = [], [], []

    with torch.no_grad():
        for orig, adv, label in data_loader:
            orig  = orig.cuda(non_blocking=True)
            adv   = adv.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            orig_pred, adv_pred, feat_loss = net(orig, adv)

            orig_accs.append((orig_pred.argmax(1) == label).float().mean().item())
            adv_accs.append((adv_pred.argmax(1) == label).float().mean().item())
            # ensure scalar for logging
            if feat_loss.dim() != 0:
                loss_val = feat_loss.mean().item()
            else:
                loss_val = feat_loss.item()
            losses.append(loss_val)

    orig_acc = float(np.mean(orig_accs))
    adv_acc  = float(np.mean(adv_accs))
    avg_loss = float(np.mean(losses))

    print(f"[VAL] Epoch {epoch:3d} — orig_acc {orig_acc:.3f}, adv_acc {adv_acc:.3f}, loss {avg_loss:.5f}")

    return orig_acc, adv_acc, avg_loss


def test(net, test_loader, result_path):
    """
    Evaluate on test set and save predictions.

    Returns:
        clean_acc     (float): accuracy on clean inputs
        adv_acc       (float): accuracy on adversarial inputs before denoising
        denoised_acc  (float): accuracy after denoising adversarial inputs
    """
    net.eval()
    if isinstance(net, DataParallel):
        classifier = net.module.classifier
    else:
        classifier = net.classifier

    clean_correct, adv_correct, denoised_correct, total = 0, 0, 0, 0
    all_clean, all_adv, all_denoised, all_labels = [], [], [], []

    with torch.no_grad():
        for orig, adv, label in test_loader:
            orig  = orig.cuda(non_blocking=True)
            adv   = adv.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # Clean accuracy
            clean_logits = classifier(orig)
            clean_pred   = clean_logits.argmax(1)

            # Adversarial before denoising
            adv_logits = classifier(adv)
            adv_pred   = adv_logits.argmax(1)

            # After denoising
            _, denoised_logits, _ = net(orig, adv)
            denoised_pred = denoised_logits.argmax(1)

            clean_correct    += (clean_pred == label).sum().item()
            adv_correct      += (adv_pred   == label).sum().item()
            denoised_correct += (denoised_pred == label).sum().item()
            total            += label.size(0)

            all_clean.append(clean_pred.cpu().numpy())
            all_adv.append(adv_pred.cpu().numpy())
            all_denoised.append(denoised_pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    clean_acc    = clean_correct / total
    adv_acc      = adv_correct   / total
    denoised_acc = denoised_correct / total

    print(f"[TEST] Clean acc: {clean_acc:.4f}, "
          f"Adv acc: {adv_acc:.4f}, "
          f"Denoised acc: {denoised_acc:.4f}")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.savez(result_path,
             clean_pred=np.concatenate(all_clean),
             adv_pred=np.concatenate(all_adv),
             denoised_pred=np.concatenate(all_denoised),
             label=np.concatenate(all_labels))

    return clean_acc, adv_acc, denoised_acc
