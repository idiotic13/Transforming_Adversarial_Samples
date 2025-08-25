import os
import argparse
import warnings
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from util import (
    get_data, get_noisy_samples,
    normalize, train_lr, compute_roc, get_model, mahalanobis_distance, 
    get_dual_manifold_scores, evaluate_model, extract_features, get_deep_features, get_mc_uncertainties, compute_roc
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅' if torch.cuda.is_available() else '❌'} Running on {device}")


def predict_in_batches(model, X, batch_size=256):
    model = model.to(device) 
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).float().to(device)
            output = model(batch)
            preds.append(output.argmax(dim=1).cpu().numpy())
            del batch, output
            torch.cuda.empty_cache()
    return np.concatenate(preds)


def main(args):
    assert args.dataset in ['mnist', 'cifar10']
    assert args.attack in ['FGSM', 'bim-a', 'bim-b', 'PGD']

    model_path = args.model_path
    data_path = args.data_path
    assert os.path.isfile(model_path), 'Model file not found.'
    assert os.path.isfile(data_path), 'Adversarial sample file not found.'

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_subdir = os.path.join(save_dir, 'saved_results')
    os.makedirs(save_subdir, exist_ok=True)

    print('Loading the data and model...') 
    model = get_model(args.dataset).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Y_train and Y_test are assumed to be direct labels (1D arrays of integers)
    X_train, X_train_adv, Y_train, X_test, X_test_adv, Y_test = get_data(data_path)

    print('Loading noisy and adversarial samples...')
    X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, args.attack)

    for s_type, dataset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
        evaluate_model(model, dataset, Y_test, s_type, args.batch_size, device)

    with torch.no_grad():
        # preds_test = model(torch.tensor(X_test).float().to(device)).argmax(dim=1).cpu().numpy()
        preds_test = predict_in_batches(model, X_test, args.batch_size)
    inds_correct = np.where(preds_test == Y_test)[0]
    X_test = X_test[inds_correct]
    X_test_noisy = X_test_noisy[inds_correct]
    X_test_adv = X_test_adv[inds_correct]

    # Define save file paths.
    mc_save_normal = os.path.join(save_subdir, f"mc_uncerts_normal_{args.dataset}_{args.attack}.npy")
    mc_save_noisy = os.path.join(save_subdir, f"mc_uncerts_noisy_{args.dataset}_{args.attack}.npy")
    mc_save_adv = os.path.join(save_subdir, f"mc_uncerts_adv_{args.dataset}_{args.attack}.npy")
    feat_save_train = os.path.join(save_subdir, f"deep_features_train_{args.dataset}_{args.attack}.npy")
    feat_save_test_normal = os.path.join(save_subdir, f"deep_features_test_normal_{args.dataset}_{args.attack}.npy")
    feat_save_test_noisy = os.path.join(save_subdir, f"deep_features_test_noisy_{args.dataset}_{args.attack}.npy")
    feat_save_test_adv = os.path.join(save_subdir, f"deep_features_test_adv_{args.dataset}_{args.attack}.npy")

    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_uncertainties(model, X_test, device, args.batch_size, mc_runs=50, save_file=mc_save_normal)
    uncerts_noisy = get_mc_uncertainties(model, X_test_noisy, device, args.batch_size, mc_runs=50, save_file=mc_save_noisy)
    uncerts_adv = get_mc_uncertainties(model, X_test_adv, device, args.batch_size, mc_runs=50, save_file=mc_save_adv)

    print('Getting deep feature representations...')
    X_train_features = get_deep_features(model, X_train, args.dataset, args.batch_size, save_file=feat_save_train)
    X_test_normal_features = get_deep_features(model, X_test, args.dataset, args.batch_size, save_file=feat_save_test_normal)
    X_test_noisy_features = get_deep_features(model, X_test_noisy, args.dataset, args.batch_size, save_file=feat_save_test_noisy)
    X_test_adv_features = get_deep_features(model, X_test_adv, args.dataset, args.batch_size, save_file=feat_save_test_adv)

    print('Estimating clean and adversarial manifolds...')
    num_classes = len(np.unique(Y_train))
    class_inds = {i: np.where(Y_train == i)[0] for i in range(num_classes)}
    manifolds_clean = {}
    manifolds_adv = {}

    for i in range(num_classes):
        class_features = X_train_features[class_inds[i]]
        manifolds_clean[i] = {
            'mean': np.mean(class_features, axis=0),
            'cov': np.cov(class_features, rowvar=False)
        }

    # preds_test_adv = model(torch.tensor(X_test_adv).float().to(device)).argmax(dim=1).cpu().numpy()
    preds_test_adv = predict_in_batches(model, X_test_adv, args.batch_size)
    for i in range(num_classes):
        inds_class_adv = np.where(preds_test_adv == i)[0]
        if len(inds_class_adv) > 0:
            class_features_adv = X_test_adv_features[inds_class_adv]
            manifolds_adv[i] = {
                'mean': np.mean(class_features_adv, axis=0),
                'cov': np.cov(class_features_adv, rowvar=False)
            }
        else:
            manifolds_adv[i] = manifolds_clean[i]

    print('Computing dual manifold distances...')
    # preds_test_normal = model(torch.tensor(X_test).float().to(device)).argmax(dim=1).cpu().numpy()
    # preds_test_noisy = model(torch.tensor(X_test_noisy).float().to(device)).argmax(dim=1).cpu().numpy()
    preds_test_normal = predict_in_batches(model, X_test, args.batch_size)
    preds_test_noisy = predict_in_batches(model, X_test_noisy, args.batch_size)

    dual_scores_normal = get_dual_manifold_scores(X_test_normal_features, preds_test_normal, manifolds_clean, 
                                                manifolds_adv, batch_size=256, device=device)
    dual_scores_noisy = get_dual_manifold_scores(X_test_noisy_features, preds_test_noisy, manifolds_clean, 
                                                manifolds_adv, batch_size=256, device=device)
    dual_scores_adv = get_dual_manifold_scores(X_test_adv_features, preds_test_adv, manifolds_clean, manifolds_adv, 
                                                batch_size=256, device=device)

    print('Training logistic regression detector...')

    X_tr, y_tr, X_val, y_val, lr = train_lr(
        densities_pos=dual_scores_adv,
        densities_neg=np.concatenate((dual_scores_normal, dual_scores_noisy)),
        uncerts_pos=uncerts_adv,
        uncerts_neg=np.concatenate((uncerts_normal, uncerts_noisy)),
        test_size=0.3
    )

    # get validation‐set probabilities
    val_probs = lr.predict_proba(X_val)[:, 1]

    # now compute ROC‑AUC on the held‐out fold
    _, _, val_auc = compute_roc(
        probs_neg=val_probs[y_val == 0],
        probs_pos=val_probs[y_val == 1],
        attack=args.attack
    )
    print(f"Detector ROCAUC (on heldout data) for {args.dataset} with {args.attack}: {val_auc:.4f}")

    # values, labels, lr = train_lr(
    #     densities_pos=dual_scores_adv,
    #     densities_neg=np.concatenate((dual_scores_normal, dual_scores_noisy)),
    #     uncerts_pos=uncerts_adv,
    #     uncerts_neg=np.concatenate((uncerts_normal, uncerts_noisy))
    # )

    # probs = lr.predict_proba(values)[:, 1]
    # n_samples = len(X_test)
    # _, _, auc_score = compute_roc(
    #     probs_neg=probs[:2 * n_samples],
    #     probs_pos=probs[2 * n_samples:]
    # )
    # print(f'Detector ROC-AUC score (Dual Manifold): {auc_score:.4f}')

    lr_filename = os.path.join(save_dir, f'lr_model_{args.dataset}_{args.attack}.pkl')
    with open(lr_filename, 'wb') as f:
        pickle.dump(lr, f)
    print(f"Logistic regression model saved to {lr_filename}")

    manifolds = {
        'clean': manifolds_clean,
        'adv': manifolds_adv
    }
    manifolds_filename = os.path.join(save_dir, f'manifolds_{args.dataset}_{args.attack}.pkl')
    with open(manifolds_filename, 'wb') as f:
        pickle.dump(manifolds, f)
    print(f"Manifolds saved to {manifolds_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str) #cifar, mnist
    parser.add_argument('-a', '--attack', required=True, type=str)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--model_path', default='/data/cs776/detector/saved_models/mnist_classifier.pth', type=str)
    parser.add_argument('--data_path', default='/data/cs776/detector/adv_data/mnist_fgsm_adv.pt', type=str)
    parser.add_argument('--save_dir', default='/data/cs776/detector/detector_save', type=str)

    args = parser.parse_args()
    main(args)
