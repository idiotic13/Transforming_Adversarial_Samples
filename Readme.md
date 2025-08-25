# Adversarial Denoising with Feature-Guided Loss

This repository implements a deep learning pipeline for **denoising adversarial examples** using feature-aligned loss functions. The denoiser is trained to align intermediate representations between clean and adversarial inputs extracted from a **frozen pretrained classifier** on either **MNIST** or **CIFAR-10**. This repository also implements an adversarial sample detector based on the dual‐manifold approach, leveraging Monte Carlo dropout uncertainties and deep feature representations from pretrained classifiers (MNIST or CIFAR-10).

---

##  Setting Up the Environment

### 1. Python Version

- Requires Python **3.12** or higher.

### 2. Install Dependencies
- # Create and activate a conda virtual environment
   conda create -n keermada python=3.12

   # Install requirements
   pip install -r requirements.txt




## Directory Structure for the purifier:
.
├── adv_data/                       # Adversarial example datasets (.pt)
│   ├── cifar10_fgsm_adv.pt
│   ├── mnist_pgd_adv.pt
│   └── ...
├── checkpoints/                    # Saved model checkpoints & best weights
│   ├── cifar10_FGSM_latest_model.pth
│   └── cifar10_FGSM_best_model.pth
├── classifier.py                  # CNN architectures & feature extractor
├── data_loader.py                 # Loads clean + adversarial .pt datasets
├── logs/                          # Training & test logs
│   ├── cifar10_FGSM_train.out
│   └── cifar10_PGD_test.out
├── main.py                        # Entry point: train or test pipeline
├── model.py                       # Builds Net: frozen classifier + denoiser + loss
├── saved_models/
    |-- mnist_classifier.pth 
    |--cifar10_classifier.pth          # Pretrained MNIST classifier weights
├── requirements.txt               # Python dependencies
└── train.py                       # Training, validation, testing logic

## Code Structure for detector

- detector_training.py:
  - Orchestrates loading models/data, generating noisy samples, extracting features & uncertainties.
  - Computes dual‐manifold scores and trains a logistic regression detector.

- util.py:
  - **Data utilities**: get_data, get_noisy_samples.
  - **Model loader**: get_model returns dataset‐specific classifier.
  - **Feature extraction**: extract_features, get_deep_features.
  - **Uncertainty estimation**: get_mc_uncertainties.
  - **Scoring**: mahalanobis_distance, get_dual_manifold_scores.
  - **Detection training**: train_lr, compute_roc.

---


## Dataset format for purifier
- Each adversarial dataset must be stored in:
   adv_data/{dataset}_{attack}_adv.pt 
   {    dataset}: cifar10 or mnist
        {attack}: one of fgsm, pgd, bim-a, bim-b

- Each .pt file contains a dictionary:
{
  "adv_train": {
    "clean":  Tensor[N, C, H, W],
    "adv":    Tensor[N, C, H, W],
    "labels": Tensor[N]
  },
  "adv_test": {
    "clean":  Tensor[N, C, H, W],
    "adv":    Tensor[N, C, H, W],
    "labels": Tensor[N]
  }
}

- Pretrained Classifier Weights
    Place classifier weights in the saved_models directory as:
    saved_models/{dataset}_classifier.pth

- Examples:
mnist_classifier.pth
cifar10_classifier.pth

    They are automatically loaded by model.py:
    wp = f"saved_models/{dataset}_classifier.pth"




## Usage(for training classifiers)
bash
python classifier_training.py --dataset cifar10

The models will be saved in saved_model directory.

## Usage(for adversarial data generation)
For mnist dataset use this drive link : https://drive.google.com/drive/folders/1MOpH3OUdAZF1BHFwu4TTWQrNQx3s0yO4?usp=sharing
Make sure to put the data files in adv_data/
For cifar10, use the below steps:

bash
python adv_generation.py --dataset cifar10 --attack FGSM --model_path saved_models/cifar10_classifier.pth

## Usage(for detector)

Run the detector script with the desired dataset and adversarial attack type. Make changes in the train_detector.sh script to set the data paths properly. In tarin_detector.sh, change the DATASET parameter for running the model on different dataset(mnist/cifar10) :

bash
bash train_detector.sh


## Usage for purifier
Run bash files for training and testing the denoiser models:
    for cifar10, run : ./run_denoiser_cifar10.sh
    for mnist, run : ./run_denoiser_mnist.sh

## Usage(for integrated detector-purifier model)
bash
bash run_integrated_model.sh



# Output 

## Output after execution of the detector

After execution of the detector, the following files are generated under <save_dir>:

- <save_dir>/ saved_results/:
  - mc_uncerts_normal_<dataset>_<attack>.npy    
  - mc_uncerts_noisy_<dataset>_<attack>.npy
  - mc_uncerts_adv_<dataset>_<attack>.npy
  - deep_features_train_<dataset>_<attack>.npy
  - deep_features_test_normal_<dataset>_<attack>.npy
  - deep_features_test_noisy_<dataset>_<attack>.npy
  - deep_features_test_adv_<dataset>_<attack>.npy

- <save_dir>/lr_model_<dataset>_<attack>.pkl    — trained logistic regression detector
- <save_dir>/manifolds_<dataset>_<attack>.pkl  — stored clean and adversarial class manifolds
- roc_curve_<attack>.png                       — ROC curve plot

---

# #logs/ (In the logs directory):
{dataset}_{attack}_train.out       # Epoch-wise training & validation metrics
{dataset}_{attack}_test.out         # Clean, adversarial, and denoised test accuracy

# #checkpoints/
cifar10_FGSM_latest_model.pth  # Checkpoint after last epoch
cifar10_FGSM_best_model.pth    # Best model by validation adversarial accuracy

# #final_test_cifar10_FGSM.npz   # NumPy archive of final predictions (clean, adv, denoised)