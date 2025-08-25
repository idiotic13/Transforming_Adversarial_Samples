import torch
from torch.utils.data import Dataset

class AdversarialTensorDataset(Dataset):
    def __init__(self, dataset, attack, phase, transform=None):
        """
        Args:
            dataset (str): 'mnist' or 'cifar10'
            attack (str): Attack name like 'FGSM', 'PGD', 'BIM-A', etc.
            phase (str): 'train' or 'test'
            transform (callable, optional): Optional transform to apply to both images.
        """
        assert phase in ['train', 'test'], "Phase must be either 'train' or 'test'"
        self.transform = transform
        self.phase = phase

        # Load the .pt file with the adversarial data
        data = torch.load(f"adv_data/{dataset}_{attack}_adv.pt")
        subset = data[f"adv_{phase}"]

        self.clean = subset['clean']
        self.adv = subset['adv']
        self.labels = subset['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clean_img = self.clean[idx]
        adv_img = self.adv[idx]
        label = self.labels[idx]

        if self.transform:
            clean_img = self.transform(clean_img)
            adv_img = self.transform(adv_img)

        return clean_img, adv_img, label
