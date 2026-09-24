import math

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL = "microsoft/rad-dino-maira-2"


# ==============================
# model
# ==============================
class Net(nn.Module):
    def __init__(self, n_classes, drop_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL, drop_path_rate=drop_path)
        self.head = nn.Linear(self.backbone.config.hidden_size, n_classes)

    def forward(self, x):
        return self.head(self.backbone(pixel_values=x).pooler_output)  # CLS token


# ==============================
# optimizer + scheduler
# ==============================
def layer_depth(name, top):
    """0 for embeddings, i+1 for transformer block i, `top` for the final norm."""
    if name.startswith("embeddings"):
        return 0
    if name.startswith("encoder.layer."):
        return int(name.split(".")[2]) + 1
    return top


def make_optimizer(net, lr_head, lr_backbone, freeze_backbone, llrd=1.0, weight_decay=0.05):
    """AdamW without weight decay on biases, norms and embedding tokens. With llrd < 1 a block at depth d
    (embeddings 0 ... final norm `top`) gets lr_backbone * llrd ** (top - d), so early layers move least.
    Param groups carry a "name" ("head" / "backbone") that the training loop uses for logging."""
    no_decay = lambda n, p: p.ndim <= 1 or any(k in n for k in ("cls_token", "mask_token", "position_embeddings"))
    buckets = {}  # (name, lr, decay) -> params
    for n, p in net.head.named_parameters():
        buckets.setdefault(("head", lr_head, not no_decay(n, p)), []).append(p)
    if freeze_backbone:
        net.backbone.requires_grad_(False)
    else:
        top = len(net.backbone.encoder.layer) + 1
        for n, p in net.backbone.named_parameters():
            lr = lr_backbone * llrd ** (top - layer_depth(n, top))
            buckets.setdefault(("backbone", lr, not no_decay(n, p)), []).append(p)
    return torch.optim.AdamW([{"name": name, "lr": lr, "weight_decay": weight_decay if decay else 0.0, "params": ps}
                              for (name, lr, decay), ps in buckets.items()])


def make_scheduler(opt, total_steps):
    """5% linear warmup, then cosine decay."""
    warm = max(1, int(0.05 * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * s / total_steps))))
