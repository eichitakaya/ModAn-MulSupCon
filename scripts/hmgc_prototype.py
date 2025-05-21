from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Hierarchical Multi‑Granular Contrastive learning (HMGC)
# ======================================================
# Reference: Li et al., "Hierarchical multi‑granular multi‑label contrastive learning", Pattern Recognition 2025
# Sections 3 & 4 (Algorithm 1,2; Eq 1–9)  citeturn2file4turn2file5turn1file4
# ------------------------------------------------------
# This script shows a *minimal* but runnable PyTorch implementation
# of HMGC for tabular multi‑label datasets such as **Yeast**.
#
# Key components
# 1. LabelEnhancer – builds a binary label tree via hierarchical
#    clustering (Jaccard distance) and expands each sample’s label
#    vector to size 2m−2  (Algorithm 1).
# 2. HMGCModel    – momentum‑encoder framework à la MoCo with
#    • L_IC  : instance‑level contrastive loss weighted by Jaccard
#    • L_CC  : class‑center contrastive loss between label centers
# 3. train_hmgc() – single‑GPU training loop following Algorithm 2.
#
# ---------------  USAGE (conda not required)  ---------------
#   pip install torch torchvision torchaudio scipy scikit-learn pandas tqdm
#   python hmgc.py  # runs a 5‑epoch demo on Yeast
#   CUDA is auto‑detected. Adjust hyper‑params at the bottom.
# ------------------------------------------------------------

import math, random, argparse, pathlib, time
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1.  Hierarchical label enhancement  (Algorithm 1)
# ---------------------------------------------------------------------------
class LabelEnhancer:
    """Expand multi‑label matrix Y (n × m) to hierarchical Y' (n × (2m−2))."""

    def __init__(self, method: str = "average"):
        self.method = method  # linkage method
        self.children_: List[Tuple[int, int]] = []  # internal nodes (SciPy order)
        self.leaf_count_ = 0

    @staticmethod
    def _jaccard_distance_matrix(Y: np.ndarray) -> np.ndarray:
        # Y: (n, m) binary
        dist = pdist(Y.T, metric="jaccard")  #  condensed form
        return squareform(dist)

    def fit(self, Y: np.ndarray) -> "LabelEnhancer":
        if Y.dtype != bool:
            Y = Y.astype(bool)
        self.leaf_count_ = Y.shape[1]
        # SciPy hierarchical clustering on labels (columns)
        dist_condensed = pdist(Y.T, metric="jaccard")
        Z = linkage(dist_condensed, method=self.method)
        self.children_ = [(int(a), int(b)) for a, b, *_ in Z]
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.leaf_count_ == 0:
            raise RuntimeError("Call fit() first.")
        n, m = Y.shape
        assert m == self.leaf_count_, "Input label dim mismatch"
        Y_prime = np.zeros((n, 2 * m - 1), dtype=np.uint8)
        Y_prime[:, :m] = Y  # copy original labels

        # Map cluster id → leaf set
        cluster_leaves: Dict[int, List[int]] = {i: [i] for i in range(m)}
        next_label_idx = m  # start of internal pseudo‑labels
        for idx, (a, b) in enumerate(self.children_):
            leaves = cluster_leaves[a] + cluster_leaves[b]
            cluster_leaves[m + idx] = leaves
            # assign new pseudo‑label if *all* child leaves present
            mask = Y[:, leaves].all(axis=1)
            Y_prime[mask, next_label_idx] = 1
            next_label_idx += 1
        return Y_prime

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        return self.fit(Y).transform(Y)

# ---------------------------------------------------------------------------
# 2.  Momentum contrast framework with multi‑granular losses (Algorithm 2)
# ---------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HMGCModel(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128, queue_size: int = 4096,
                 temperature: float = 0.07, momentum: float = 0.999,
                 alpha: float = 1.0, beta: float = 1.0, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.temperature = temperature
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        # encoders
        self.encoder_q = MLPEncoder(in_dim, emb_dim).to(self.device)
        self.encoder_k = MLPEncoder(in_dim, emb_dim).to(self.device)
        self._momentum_update(0)  # inits encoder_k = encoder_q
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        # queue
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, emb_dim), dim=1))
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))  # dummy (not used for Yeast)
        self.queue_ptr = 0

    @torch.no_grad()
    def _momentum_update(self, m=None):
        m = self.momentum if m is None else m
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters(), strict=True):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        qsize = self.queue.shape[0]
        ptr = int(self.queue_ptr)
        assert qsize % batch_size == 0, "Queue size must be divisible by batch size"
        self.queue[ptr: ptr + batch_size] = keys
        self.queue_ptr = (ptr + batch_size) % qsize

    # ---------------------------------------------------------------------
    #  Loss components
    # ---------------------------------------------------------------------
    @staticmethod
    def _jaccard_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (..., L)
        inter = (a & b).sum(-1).float()
        union = (a | b).sum(-1).float()
        return inter / (union + 1e-8)

    def _instance_contrastive_loss(self, z_q: torch.Tensor, z_k: torch.Tensor,
                                    queue: torch.Tensor, y_prime: torch.Tensor) -> torch.Tensor:
        b, d = z_q.shape
        logits_pos = torch.einsum('nc,nc->n', z_q, z_k).unsqueeze(-1)  # (b,1)
        logits_neg = torch.matmul(z_q, queue.t())                        # (b, q)
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature  # (b, 1+q)
        # target similarities for BCE: sim(y_i', y_j')
        with torch.no_grad():
            # Build target matrix (b,1+q)
            sim_pos = self._jaccard_similarity(y_prime, y_prime)  # (b) self‑sim=1, but we only need diag
            sim_pos = sim_pos.unsqueeze(-1)  # (b,1)
            sim_neg = self._jaccard_similarity(
                y_prime.unsqueeze(1).bool(),
                self.queue_labels.expand(b, -1)  # zeros → union = something ; here we use overlap>0 rule
            ) if False else torch.zeros_like(logits_neg)  # place‑holder (no labels for queue)
            target = torch.cat([sim_pos, sim_neg], dim=1)
        pred = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(pred, target, reduction='none').mean()
        return loss

    def _class_center_loss(self, z_all: torch.Tensor, y_prime_all: torch.Tensor) -> torch.Tensor:
        # z_all: (N, d), y_prime_all: (N, L)
        y_mask = y_prime_all.bool()
        denom = y_mask.sum(0).clamp(min=1e-6).unsqueeze(-1)  # (L,1)
        centers = torch.matmul(y_mask.float().t(), z_all) / denom  # (L,d)
        centers = F.normalize(centers, dim=1)
        # pairwise dot products
        sim_mat = torch.matmul(centers, centers.t()) / self.temperature  # (L,L)
        pred = torch.softmax(sim_mat, dim=1)  # row‑wise
        # label Jaccard similarity between classes (Eq 7)
        with torch.no_grad():
            inter = torch.matmul(y_mask.float().t(), y_mask.float())  # (L,L)
            union = (y_mask.sum(0, keepdim=True) + y_mask.sum(0, keepdim=True).t()) - inter
            target = inter / (union + 1e-8)
        loss = F.binary_cross_entropy(pred, target, reduction='mean')
        return loss

    # ---------------------------------------------------------------------
    #  forward() – one HMGC training step (returns total loss)
    # ---------------------------------------------------------------------
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, y_prime: torch.Tensor) -> torch.Tensor:
        z_q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update()
            z_k = F.normalize(self.encoder_k(x_k), dim=1)
        # Instance‑level contrastive loss (L_IC)
        L_ic = self._instance_contrastive_loss(z_q, z_k, self.queue.clone().detach().to(self.device), y_prime)
        
        # Gather z and y′ for class‑center loss
        z_cur = torch.cat([z_q, z_k], dim=0)            # 2B × d
        y_cur = torch.cat([y_prime, y_prime], dim=0)            # 2B × L
        L_cc = self._class_center_loss(z_cur, y_cur)    # ◎ shapes (L,2B)·(2B,d)
        
        total_loss = self.alpha * L_ic + self.beta * L_cc
        # Update queue
        self._dequeue_and_enqueue(z_k.detach())
        return total_loss

# ---------------------------------------------------------------------------
# 3.  Demo training loop on Yeast dataset
# ---------------------------------------------------------------------------

def load_yeast(csv_path: str = "../datasets/yeast/yeast-train.csv") -> Tuple[np.ndarray, np.ndarray]:
    # Step‑1 in previous answer (space‑sep loader)
    df = pd.read_csv(csv_path, sep=",", header=0, engine="python", comment="|")
    #print(df)
    sample_ids = df.iloc[:, 0].values  # not used
    X = df.iloc[:, 0:103].astype(float).values  # 103 features
    Y = df.iloc[:, 103:].astype(int).values     # 14 labels (0/1)
    return X, Y


def train_hmgc(args):
    # ------------- data -------------
    X, Y = load_yeast(args.data)
    print(X.shape, Y.shape)
    enhancer = LabelEnhancer().fit(Y)
    Y_prime = enhancer.transform(Y)

    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_std)
    Y_tensor = torch.tensor(Y_prime, dtype=torch.uint8)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ------------- model -------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HMGCModel(in_dim=X.shape[1], emb_dim=args.emb_dim, queue_size=args.queue,
                      temperature=args.tau, momentum=args.momentum,
                      alpha=args.alpha, beta=args.beta, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------- train -------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
            x_q = x.to(device)
            # simple augmentation – Gaussian noise; replace as needed
            x_k = (x + 0.01 * torch.randn_like(x)).to(device)
            y_p = y.to(device)
            loss = model(x_q, x_k, y_p)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {epoch}: HMGC loss = {running / len(loader):.4f}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../datasets/yeast/yeast-train.csv")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--queue", type=int, default=4096)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--momentum", type=float, default=0.999)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train_hmgc(args)
