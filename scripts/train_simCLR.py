"""
SimCLR ‑ Pure PyTorch Implementation
===================================
A *single‑file* baseline you can copy‑paste and run without any external SSL
frameworks (no Lightning, no VISSL).  It covers:
  • ResNet‑50 encoder  (swapable)
  • 2‑layer projection MLP
  • NT‑Xent (InfoNCE) loss with temperature
  • Two‑view augmentation pipeline from the original SimCLR paper
  • Mixed‑precision & single‑process multi‑GPU (DDP) optional
  • Checkpointing & resume

Run example (single GPU, ImageNet‑100 subset):
    python simclr_pretrain.py \
        --data_dir /path/to/imagenet100 \
        --epochs 200 --batch_size 256 --lr 0.5 \
        --temperature 0.1

Multi‑GPU (8 gpus, FP16) with torchrun:
    torchrun --nproc_per_node 8 simclr_pretrain.py \
        --data_dir /path/to/imagenet \
        --epochs 200 --batch_size 256 \
        --lr 0.5 --temperature 0.1 --fp16
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import datasets

# -----------------------
# 1. Data Augmentations
# -----------------------
class SimCLRTransform:
    """Apply 2 stochastic views of the same image."""
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1*image_size)+1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# --------------------------------------
# 2. NT-Xent Loss (InfoNCE with 2N batch)
# --------------------------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, world_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._create_mask(batch_size*world_size)

    @staticmethod
    def _create_mask(num_samples):
        diag = torch.eye(num_samples, dtype=torch.bool)
        l1 = diag.roll(num_samples//2, dims=0)
        mask = (~diag) & (~l1)
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (N, D)
        """
        if dist.is_initialized():
            z_i = torch.cat(torch.distributed.nn.all_gather(z_i), dim=0)
            z_j = torch.cat(torch.distributed.nn.all_gather(z_j), dim=0)

        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        sim = torch.matmul(z, z.t()) / self.temperature  # (2N,2N)
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        negatives = sim[self.mask.repeat(2, 2)]  # remove i==j and positives
        negatives = negatives.view(2*N, -1)

        labels = torch.zeros(2*N, dtype=torch.long, device=z.device)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss /= (2*N)
        return loss

# -----------------------------
# 3. SimCLR Model Architecture
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, base_encoder="resnet18", proj_hidden_dim=2048, proj_out_dim=128):
        super().__init__()
        backbone = models.__dict__[base_encoder](weights=None)
        # 1チャンネル対応に変更
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(num_ftrs, proj_hidden_dim, proj_out_dim)

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        return z1, z2

# ----------------
# 4. Train script
# ----------------

def main():
    parser = argparse.ArgumentParser(description="SimCLR Pretraining ‑ Pure PyTorch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)
    parser.add_argument("--proj_out_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_dir", default="../experiment/simCLR")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = SimCLRTransform()
    dataset = datasets.RadImageNetDataset(transform=transform)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    model = SimCLR(proj_hidden_dim=args.proj_hidden_dim, proj_out_dim=args.proj_out_dim)
    model = model.to(device)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    criterion = NTXentLoss(batch_size=args.batch_size, temperature=args.temperature)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    z1, z2 = model(x1, x2)
                    loss = criterion(z1, z2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1, z2 = model(x1, x2)
                loss = criterion(z1, z2)
                #print(loss)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()*x1.size(0)
        scheduler.step()
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}")

        # --- checkpoint ---
        if epoch == args.epochs - 1:  # 最終エポックのみ保存
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = Path(args.save_dir) / f"simclr_final.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)

if __name__ == "__main__":
    main()
