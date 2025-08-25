import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.resnet import BasicBlock
from classifier import CIFAR10Classifier, MNISTClassifier

# -------------------------------------------------------------------
# 1) Config dict for the pipeline
# -------------------------------------------------------------------
base_config = {
    'flip': True,
    'net_type': 'resnet32',
    'dataset': None,
    'loss_idcs': None  # will be set per dataset
}

# -------------------------------------------------------------------
# 2) DenoiseLoss
# -------------------------------------------------------------------
class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super().__init__()
        assert 0 <= hard_mining <= 1
        self.n = n
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.abs(x - y).pow(self.n).div(self.n)
        if self.hard_mining > 0:
            flat = loss.view(-1)
            k = int(flat.numel() * self.hard_mining)
            flat, idx = torch.topk(flat, k)
            y = y.view(-1)[idx]
            loss = flat
        loss = loss.mean()
        if self.norm:
            norm_term = torch.abs(y).pow(self.n).mean().item()
            loss = loss / norm_term
        return loss

class Loss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super().__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)

    def forward(self, x_dict, y_dict, loss_keys=None):
        if loss_keys is None:
            loss_keys = x_dict.keys()
        return [self.loss(x_dict[k], y_dict[k]) for k in loss_keys]

# -------------------------------------------------------------------
# 3) Denoiser Module
# -------------------------------------------------------------------
class Denoise(nn.Module):
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back):
        super().__init__()
        h, w = [], []
        for _ in num_fwd:
            h.append(h_in); w.append(w_in)
            h_in = int(np.ceil(h_in/2)); w_in = int(np.ceil(w_in/2))

        expansion = 1
        fwd, back, upsample = [], [], []
        n_in = fwd_in

        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                stride = 1 if (i == 0) else 2 if j == 0 else 1
                downsample = None
                if stride != 1 or n_in != fwd_out[i]:
                    downsample = nn.Sequential(
                        nn.Conv2d(n_in, fwd_out[i], kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(fwd_out[i])
                    )
                group.append(block(n_in, fwd_out[i], stride=stride, downsample=downsample))
                n_in = fwd_out[i]
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in reversed(range(len(num_back))):
            upsample.insert(0, nn.Upsample(size=(h[i], w[i]), mode='bilinear'))
            group = []
            for j in range(num_back[i]):
                downsample = None
                if j == 0 and n_in != back_out[i]:
                    downsample = nn.Sequential(
                        nn.Conv2d(n_in, back_out[i], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(back_out[i])
                    )
                group.append(block(n_in, back_out[i], downsample=downsample))
                n_in = back_out[i]
            if i > 0:
                n_in = (back_out[i] + fwd_out[i-1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back     = nn.ModuleList(back)

        self.residual_proj = nn.Conv2d(fwd_in, fwd_in, kernel_size=1, bias=False)
        self.final = nn.Conv2d(back_out[0]*expansion, fwd_in, kernel_size=1, bias=False)

    def forward(self, x):
        out = x
        skips = []
        for i, layer in enumerate(self.fwd):
            out = layer(out)
            if i != len(self.fwd) - 1:
                skips.append(out)
        for i in reversed(range(len(self.back))):
            out = self.upsample[i](out)
            out = torch.cat([out, skips[i]], dim=1)
            out = self.back[i](out)
        if hasattr(self, 'final'):
            out = self.final(out)
        residual = self.residual_proj(x)
        return out + residual

# -------------------------------------------------------------------
# 4) Net = Denoise + Classifier
# -------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, dataset='cifar10', feat_layer='pre_fc2'):
        super().__init__()
        self.dataset = dataset
        self.loss_idcs = feat_layer

        if dataset == 'cifar10':
            self.denoise = Denoise(
                h_in=32, w_in=32, block=BasicBlock, fwd_in=3,
                fwd_out=[64, 128, 256, 256, 256], num_fwd=[2, 3, 3, 3, 3],
                back_out=[64, 128, 256, 256], num_back=[2, 3, 3, 3]
            )
            self.classifier = CIFAR10Classifier()
        elif dataset == 'mnist':
            self.denoise = Denoise(
                h_in=28, w_in=28, block=BasicBlock, fwd_in=1,
                fwd_out=[32, 64, 128], num_fwd=[1, 2, 2],
                back_out=[32, 64], num_back=[1, 2]
            )
            self.classifier = MNISTClassifier()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        wp = f"saved_models/{dataset}_classifier.pth"
        if os.path.exists(wp):
            sd = torch.load(wp, map_location="cpu")
            new_sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
            self.classifier.load_state_dict(new_sd)

        for p in self.classifier.parameters():
            p.requires_grad = False

        self.guided_loss = Loss(n=1, hard_mining=0, norm=False)

    def forward(self, orig_x, adv_x, requires_control=False):
        layer = self.loss_idcs

        orig_feats = {layer: self.classifier._extract_feats(orig_x, layer=layer)}
        orig_logits = self.classifier(orig_x)

        if requires_control:
            ctrl_feats = {layer: self.classifier._extract_feats(adv_x, layer=layer)}
            ctrl_logits = self.classifier(adv_x)
            ctrl_loss = self.guided_loss(ctrl_feats, orig_feats, [layer])

        den = self.denoise(adv_x)
        adv_feats = {layer: self.classifier._extract_feats(den, layer=layer)}
        adv_logits = self.classifier(den)
        adv_loss = sum(self.guided_loss(adv_feats, orig_feats, [layer]))

        if requires_control:
            return orig_logits, adv_logits, adv_loss, ctrl_logits, ctrl_loss
        else:
            return orig_logits, adv_logits, adv_loss

# -------------------------------------------------------------------
# 5) get_model()
# -------------------------------------------------------------------
def get_model(dataset, feat_layer=None):
    config = base_config.copy()
    config['dataset'] = dataset

    # Dataset-specific feature layer for loss
    if dataset == 'cifar10':
        config['loss_idcs'] = 'pre_fc2' if feat_layer is None else feat_layer
    elif dataset == 'mnist':
        config['loss_idcs'] = 'flat' if feat_layer is None else feat_layer
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return config, Net(dataset=dataset, feat_layer=config['loss_idcs'])
