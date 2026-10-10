#!/usr/bin/env python3
"""
train_segmenter.py

Phase-1 standalone segmenter that distils the reviewed SAM3 masks into a
promptless, full-resolution, class-agnostic RAT vs BACKGROUND model.

WHY BINARY (not obj1/obj2)
--------------------------
The two rats are visually near-identical; identity is temporal, not
appearance-based, so a network cannot learn "rat1 vs rat2". The hard,
valuable problem is producing a COMPLETE, pixel-accurate mask of where the
rats are - recovering the shadowed/occluded tail that SAM3 drops. This model
targets exactly that. Splitting the predicted union into obj1/obj2 with stable
identity is a separate downstream step (connected components + your existing
temporal re-id); a minimal splitter is provided (split_into_instances) but is
NOT part of training.

INPUTS  (per frame, built on the fly - no giant precomputed dataset)
------
  - RGB frame          : decoded from the source .mp4 via decord (frame-exact)
  - foreground prior    : shadow-suppressed background-subtraction channel,
                          computed from the cached per-session background using
                          prepare_dataset.compute_foreground_prior
  => 4-channel input (R,G,B,prior)

TARGET
------
  binary mask = (label >= 1)   from masks_raw palette PNG (values 0/1/2)

MODEL
-----
  U-Net with an ImageNet-pretrained ResNet-34 encoder; conv1 adapted 3->4 ch
  (pretrained RGB weights kept, 4th channel zero-init so training learns to use
  the prior). Full-resolution single-channel logit output.

LOSS
----
  BCE + Tversky (alpha<beta) to bias toward RECALL - we would rather slightly
  over-include a faint tail than drop it, which is the whole point.

DEFAULTS target the RTX 4500 Ada (24 GB, GPU 0). Train at ~0.53x scale
(640x1024, aspect preserved, both /32); eval can run at full res via pad-to-32.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision

# reuse the EXACT foreground-prior implementation from the data-prep step
from prepare_dataset import compute_foreground_prior

from decord import VideoReader, cpu as decord_cpu


# ===========================================================================
# Data
# ===========================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _round32(x: int) -> int:
    return int(math.ceil(x / 32.0) * 32)


class _VideoReaderCache:
    """Per-worker LRU of decord readers (decord readers are not fork-safe, so
    we open lazily inside the worker process)."""

    def __init__(self, maxsize: int = 3):
        self.maxsize = maxsize
        self._d: "dict[str, VideoReader]" = {}
        self._order: List[str] = []

    def get(self, path: str) -> VideoReader:
        if path in self._d:
            self._order.remove(path)
            self._order.append(path)
            return self._d[path]
        vr = VideoReader(path, ctx=decord_cpu(0))
        self._d[path] = vr
        self._order.append(path)
        if len(self._order) > self.maxsize:
            old = self._order.pop(0)
            self._d.pop(old, None)
        return vr


def load_manifest(manifest_path: str, splits: Tuple[str, ...]) -> List[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] in splits:
                rows.append(r)
    return rows


@dataclass
class DataConfig:
    train_h: int = 640
    train_w: int = 1024
    full_res: bool = False          # eval at full resolution (pad to /32)
    augment: bool = True
    flip_p: float = 0.5
    photometric_p: float = 0.7
    video_cache_size: int = 3       # open decord readers per worker; keep small
                                    # (each holds decode buffers for a 1080p+ video)


class RatSegDataset(Dataset):
    def __init__(self, rows: List[dict], cfg: DataConfig, train: bool):
        self.rows = rows
        self.cfg = cfg
        self.train = train
        self._bg_cache: Dict[str, np.ndarray] = {}
        self._vr_cache = _VideoReaderCache(maxsize=cfg.video_cache_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _bg(self, path: str) -> np.ndarray:
        if path not in self._bg_cache:
            bg = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
            if bg is None:
                raise RuntimeError(f"Could not read background: {path}")
            self._bg_cache[path] = bg
        return self._bg_cache[path]

    def _load_raw(self, row: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vr = self._vr_cache.get(row["video_path"])
        idx = int(row["frame_idx"])
        frame_rgb = vr[idx].asnumpy()  # HxWx3 RGB uint8
        bg_bgr = self._bg(row["background_path"])
        label = np.array(Image.open(row["mask_path"]))
        if label.ndim == 3:
            label = label[..., 0]
        return frame_rgb, bg_bgr, label

    def _photometric(self, rgb: np.ndarray) -> np.ndarray:
        # brightness/contrast/gamma jitter on RGB only (lighting-epoch robustness)
        img = rgb.astype(np.float32) / 255.0
        if random.random() < 0.8:
            b = random.uniform(-0.12, 0.12)
            c = random.uniform(0.85, 1.15)
            img = np.clip((img - 0.5) * c + 0.5 + b, 0, 1)
        if random.random() < 0.5:
            g = random.uniform(0.8, 1.25)
            img = np.clip(img ** g, 0, 1)
        return (img * 255.0).astype(np.uint8)

    def __getitem__(self, i: int):
        row = self.rows[i]
        frame_rgb, bg_bgr, label = self._load_raw(row)

        # foreground prior at native res (needs BGR to match prepare_dataset)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        prior = compute_foreground_prior(frame_bgr, bg_bgr)   # HxW float32 [0,1]

        target = (label >= 1).astype(np.float32)              # binary union mask

        # ---- resize ----
        if self.cfg.full_res:
            H = _round32(frame_rgb.shape[0])
            W = _round32(frame_rgb.shape[1])
        else:
            H, W = self.cfg.train_h, self.cfg.train_w

        interp_img = cv2.INTER_AREA if (H < frame_rgb.shape[0]) else cv2.INTER_LINEAR
        rgb = cv2.resize(frame_rgb, (W, H), interpolation=interp_img)
        prior_r = cv2.resize(prior, (W, H), interpolation=cv2.INTER_LINEAR)
        target_r = cv2.resize(target, (W, H), interpolation=cv2.INTER_NEAREST)

        # ---- augment (train only) ----
        if self.train and self.cfg.augment:
            if random.random() < self.cfg.flip_p:
                rgb = rgb[:, ::-1].copy()
                prior_r = prior_r[:, ::-1].copy()
                target_r = target_r[:, ::-1].copy()
            if random.random() < self.cfg.photometric_p:
                rgb = self._photometric(rgb)

        # ---- to tensor + normalize ----
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        img = np.concatenate([rgb_f, prior_r[..., None]], axis=2)  # HxWx4
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        tgt_t = torch.from_numpy(target_r[None]).float()
        return img_t, tgt_t


# ===========================================================================
# Model : ResNet-34 U-Net, 4-channel input
# ===========================================================================

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNet34UNet(nn.Module):
    """U-Net with a torchvision ResNet-34 encoder adapted to 4 input channels."""

    def __init__(self, in_ch: int = 4, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        enc = torchvision.models.resnet34(weights=weights)

        # adapt conv1: 3 -> in_ch, keep pretrained RGB weights, zero-init extras
        old = enc.conv1
        new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                new_conv.weight[:, :3] = old.weight
                if in_ch > 3:
                    new_conv.weight[:, 3:].zero_()
            enc.conv1 = new_conv

        self.input = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # -> 64, /2
        self.pool = enc.maxpool                                   # -> /4
        self.layer1 = enc.layer1   # 64,  /4
        self.layer2 = enc.layer2   # 128, /8
        self.layer3 = enc.layer3   # 256, /16
        self.layer4 = enc.layer4   # 512, /32

        self.up4 = ConvBlock(512 + 256, 256)
        self.up3 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)
        self.up1 = ConvBlock(64 + 64, 64)
        self.up0 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0 = self.input(x)     # 64,  /2
        x1 = self.layer1(self.pool(x0))  # 64,  /4
        x2 = self.layer2(x1)   # 128, /8
        x3 = self.layer3(x2)   # 256, /16
        x4 = self.layer4(x3)   # 512, /32

        d4 = self.up4(torch.cat([self._up(x4, x3), x3], 1))
        d3 = self.up3(torch.cat([self._up(d4, x2), x2], 1))
        d2 = self.up2(torch.cat([self._up(d3, x1), x1], 1))
        d1 = self.up1(torch.cat([self._up(d2, x0), x0], 1))
        d0 = self.up0(F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(d0)   # logits, full input res, 1 channel


# ===========================================================================
# Losses & metrics
# ===========================================================================

def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    """alpha weights FP, beta weights FN. beta>alpha => penalise misses (recall)."""
    p = torch.sigmoid(logits)
    p = p.view(p.size(0), -1)
    t = target.view(target.size(0), -1)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tv).mean()


class SegLoss(nn.Module):
    def __init__(self, bce_w=0.5, tv_w=0.5, alpha=0.3, beta=0.7, pos_weight=None):
        super().__init__()
        self.bce_w, self.tv_w = bce_w, tv_w
        self.alpha, self.beta = alpha, beta
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        tv = tversky_loss(logits, target, self.alpha, self.beta)
        return self.bce_w * bce + self.tv_w * tv, bce.detach(), tv.detach()


@torch.no_grad()
def iou_dice_recall(logits, target, thresh=0.5):
    p = (torch.sigmoid(logits) > thresh).float()
    t = (target > 0.5).float()
    inter = (p * t).sum((1, 2, 3))
    union = ((p + t) > 0).float().sum((1, 2, 3))
    tp = (p * t).sum((1, 2, 3))
    iou = (inter + 1) / (union + 1)
    dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    recall = (tp + 1) / (t.sum((1, 2, 3)) + 1)
    return iou.mean().item(), dice.mean().item(), recall.mean().item()


# ===========================================================================
# Sampler : balance sessions so long ones don't dominate
# ===========================================================================

def session_weighted_sampler(rows: List[dict]) -> WeightedRandomSampler:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["experiment"]] = counts.get(r["experiment"], 0) + 1
    weights = [1.0 / counts[r["experiment"]] for r in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


# ===========================================================================
# Inference utilities (NOT used in training)
# ===========================================================================

@torch.no_grad()
def predict_union_mask(model, frame_rgb, bg_bgr, device, thresh=0.5) -> np.ndarray:
    """Full-res binary rat mask for one frame. frame_rgb: HxWx3 RGB uint8."""
    model.eval()
    H0, W0 = frame_rgb.shape[:2]
    H, W = _round32(H0), _round32(W0)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    prior = compute_foreground_prior(frame_bgr, bg_bgr)
    rgb = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    pr = cv2.resize(prior, (W, H), interpolation=cv2.INTER_LINEAR)
    img = np.concatenate([rgb, pr[..., None]], 2).transpose(2, 0, 1)[None]
    x = torch.from_numpy(img).float().to(device)
    logit = model(x)
    m = (torch.sigmoid(logit)[0, 0].cpu().numpy() > thresh).astype(np.uint8)
    return cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)


def split_into_instances(union_mask: np.ndarray, prev_masks=None) -> Dict[int, np.ndarray]:
    """
    Minimal per-frame split of the union into up to two instances by taking the
    two largest connected components. Identity (which is obj1 vs obj2) should be
    resolved by matching to prev_masks with the existing temporal re-id; here we
    just do IoU matching if prev_masks is given, else area order.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(union_mask.astype(np.uint8), 8)
    comps = sorted(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_AREA], reverse=True)[:2]
    masks = [(lbl == c) for c in comps]
    out: Dict[int, np.ndarray] = {}
    if prev_masks and masks:
        # greedy IoU assignment to previous identities
        ids = list(prev_masks.keys())
        for m in masks:
            best, bid = 0.0, None
            for k in ids:
                pm = prev_masks[k]
                inter = np.logical_and(m, pm).sum()
                uni = np.logical_or(m, pm).sum()
                iou = inter / uni if uni else 0.0
                if iou > best:
                    best, bid = iou, k
            if bid is not None:
                out[bid] = m
                ids.remove(bid)
        for k, m in zip([1, 2], masks):
            out.setdefault(k, m)
    else:
        for k, m in zip([1, 2], masks):
            out[k] = m
    return out


# ===========================================================================
# Eval
# ===========================================================================

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> Dict[str, float]:
    model.eval()
    ious, dices, recs = [], [], []
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logit = model(img)
        i, d, r = iou_dice_recall(logit.float(), tgt)
        ious.append(i); dices.append(d); recs.append(r)
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)), "recall": float(np.mean(recs))}


# ===========================================================================
# Train
# ===========================================================================

@dataclass
class TrainConfig:
    manifest: str = ""
    out_dir: str = "seg_run"
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_accum: int = 1
    device: str = "cuda:0"
    amp: str = "bf16"               # bf16 | fp16 | none
    train_h: int = 640
    train_w: int = 1024
    eval_full_res: bool = False
    pretrained: bool = True
    pos_weight: Optional[float] = None
    video_cache_size: int = 3       # open decord readers per worker
    persistent_workers: bool = False
    seed: int = 0


def _amp_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


def build_loaders(cfg: TrainConfig):
    train_rows = load_manifest(cfg.manifest, ("train",))
    val_rows = load_manifest(cfg.manifest, ("val",))

    dcfg = DataConfig(train_h=cfg.train_h, train_w=cfg.train_w, augment=True,
                      video_cache_size=cfg.video_cache_size)
    dcfg_eval = DataConfig(train_h=cfg.train_h, train_w=cfg.train_w,
                           full_res=cfg.eval_full_res, augment=False,
                           video_cache_size=cfg.video_cache_size)

    train_ds = RatSegDataset(train_rows, dcfg, train=True)
    val_ds = RatSegDataset(val_rows, dcfg_eval, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        sampler=session_weighted_sampler(train_rows),
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1 if cfg.eval_full_res else cfg.batch_size,
        shuffle=False, num_workers=max(1, cfg.num_workers // 2), pin_memory=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader, len(train_rows), len(val_rows)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")
    amp_dtype = _amp_dtype(cfg.amp) if device.type == "cuda" else None

    train_loader, val_loader, n_tr, n_va = build_loaders(cfg)
    print(f"train frames={n_tr}  val frames={n_va}  device={device}  amp={cfg.amp}")

    model = ResNet34UNet(in_ch=4, pretrained=cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16"))
    loss_fn = SegLoss(pos_weight=cfg.pos_weight)

    log_path = out / "train_log.jsonl"
    best_iou = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run = {"loss": 0.0, "bce": 0.0, "tv": 0.0, "n": 0}
        opt.zero_grad(set_to_none=True)
        for it, (img, tgt) in enumerate(train_loader):
            img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logit = model(img)
                loss, bce, tv = loss_fn(logit, tgt)
                loss = loss / cfg.grad_accum
            if cfg.amp == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (it + 1) % cfg.grad_accum == 0:
                if cfg.amp == "fp16":
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            run["loss"] += loss.item() * cfg.grad_accum; run["bce"] += bce.item()
            run["tv"] += tv.item(); run["n"] += 1
        sched.step()

        val = evaluate(model, val_loader, device, amp_dtype)
        rec = {
            "epoch": epoch,
            "train_loss": run["loss"] / max(run["n"], 1),
            "train_bce": run["bce"] / max(run["n"], 1),
            "train_tversky": run["tv"] / max(run["n"], 1),
            "val_iou": val["iou"], "val_dice": val["dice"], "val_recall": val["recall"],
            "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{epoch:03d}] loss={rec['train_loss']:.4f} "
              f"val_iou={val['iou']:.4f} val_dice={val['dice']:.4f} "
              f"val_recall={val['recall']:.4f} ({rec['sec']}s)")

        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                   out / "last.pt")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "val_iou": best_iou}, out / "best.pt")
            print(f"        new best val_iou={best_iou:.4f} -> best.pt")

    print(f"done. best val_iou={best_iou:.4f}")
    return str(out / "best.pt")


@torch.no_grad()
def run_test(cfg: TrainConfig, ckpt: str, test_split: str = "test"):
    """Evaluate a checkpoint on test rows, reported PER experiment/source."""
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")
    amp_dtype = _amp_dtype(cfg.amp) if device.type == "cuda" else None
    model = ResNet34UNet(in_ch=4, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])

    rows = load_manifest(cfg.manifest, (test_split,))
    by_exp: Dict[str, List[dict]] = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], []).append(r)

    dcfg = DataConfig(train_h=cfg.train_h, train_w=cfg.train_w,
                      full_res=cfg.eval_full_res, augment=False,
                      video_cache_size=cfg.video_cache_size)
    results = {}
    for exp, ers in by_exp.items():
        loader = DataLoader(RatSegDataset(ers, dcfg, train=False),
                            batch_size=1 if cfg.eval_full_res else cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            prefetch_factor=2 if cfg.num_workers > 0 else None)
        results[exp] = evaluate(model, loader, device, amp_dtype)
        print(f"  TEST {exp}: iou={results[exp]['iou']:.4f} "
              f"dice={results[exp]['dice']:.4f} recall={results[exp]['recall']:.4f}")
    return results


def load_model_from_ckpt(ckpt: str, device: torch.device) -> nn.Module:
    model = ResNet34UNet(in_ch=4, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    ep = state.get("epoch", "?")
    vi = state.get("val_iou", None)
    print(f"Loaded {ckpt} (epoch {ep}" + (f", val_iou {vi:.4f})" if vi is not None else ")"))
    return model


@torch.no_grad()
def sweep_threshold(cfg: "TrainConfig", ckpt: str, split: str = "val",
                    thresholds: Optional[List[float]] = None,
                    max_scan: int = 6000) -> List[dict]:
    """
    Run the split ONCE (single decode pass), accumulate per-threshold
    confusion-matrix totals in memory, and report IoU / Dice / recall /
    precision at each threshold. Micro-averaged over pixels (summing TP/FP/FN
    across all frames), which is the right aggregate for picking an operating
    point - it isn't dominated by tiny-rat frames the way per-frame mean IoU is.

    The point: your worst frames over-include cast shadow, so raising the
    threshold should trim FP (raising IoU/precision) while barely touching
    recall (tails stay). This finds the knee.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")
    model = load_model_from_ckpt(ckpt, device)

    rows = load_manifest(cfg.manifest, (split,))
    if len(rows) > max_scan:
        rng0 = random.Random(0)
        rows = [rows[i] for i in sorted(rng0.sample(range(len(rows)), max_scan))]
        print(f"Split '{split}': scoring a random {max_scan}-frame subset.")
    else:
        print(f"Split '{split}': scoring all {len(rows)} frames.")

    dcfg = DataConfig(train_h=cfg.train_h, train_w=cfg.train_w,
                      full_res=cfg.eval_full_res, augment=False,
                      video_cache_size=cfg.video_cache_size)
    loader = DataLoader(RatSegDataset(rows, dcfg, train=False),
                        batch_size=1 if cfg.eval_full_res else cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        prefetch_factor=2 if cfg.num_workers > 0 else None)
    amp_dtype = _amp_dtype(cfg.amp) if device.type == "cuda" else None

    thr = torch.tensor(thresholds, dtype=torch.float32)
    TP = torch.zeros(len(thresholds))
    FP = torch.zeros(len(thresholds))
    FN = torch.zeros(len(thresholds))

    n = 0
    for img, tgt in loader:
        img = img.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logit = model(img)
        prob = torch.sigmoid(logit.float()).cpu()          # B,1,h,w
        t = (tgt > 0.5).float().cpu()                       # B,1,h,w
        # broadcast prob over thresholds: for each t_k, count TP/FP/FN
        for k, tv in enumerate(thresholds):
            p = (prob > tv).float()
            TP[k] += (p * t).sum().item()
            FP[k] += (p * (1 - t)).sum().item()
            FN[k] += ((1 - p) * t).sum().item()
        n += img.shape[0]

    print(f"\n=== THRESHOLD SWEEP ({split}, {n} frames, micro-averaged) ===")
    print(f"{'thresh':>7} {'IoU':>8} {'Dice':>8} {'recall':>8} {'precision':>10}")
    results = []
    for k, tv in enumerate(thresholds):
        tp, fp, fn = TP[k].item(), FP[k].item(), FN[k].item()
        iou = tp / (tp + fp + fn + 1e-9)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        results.append({"thresh": tv, "iou": iou, "dice": dice,
                        "recall": rec, "precision": prec})
        print(f"{tv:>7.2f} {iou:>8.4f} {dice:>8.4f} {rec:>8.4f} {prec:>10.4f}")

    best = max(results, key=lambda r: r["iou"])
    print(f"\nBest IoU at threshold {best['thresh']:.2f}: "
          f"IoU={best['iou']:.4f} recall={best['recall']:.4f} precision={best['precision']:.4f}")
    print("Pick the knee: highest threshold that keeps recall high while IoU/precision climb.")

    out = Path(cfg.out_dir) / f"threshold_sweep_{split}.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")
    return results


@torch.no_grad()
def dump_overlays(cfg: "TrainConfig", ckpt: str, split: str, n_worst: int = 40,
                  n_random: int = 20, out_subdir: str = "overlays",
                  thresh: float = 0.5, max_scan: int = 6000) -> None:
    """
    Write predicted-mask-on-frame PNGs for the N worst-IoU frames (where the
    model struggles most - the frames you actually need to eyeball) plus some
    random ones, at FULL resolution.

    Each PNG is a side-by-side triptych:
        [ frame ] [ ground-truth mask overlaid ] [ prediction overlaid ]
    GT in green, prediction in red; where they agree you see yellow. Missing
    tail = green with no red over it; hallucinated pixels = red with no green.

    max_scan: to find the worst frames we must score frames first; on a huge
    split (e.g. the 135k-frame test set) scoring ALL of them single-process is
    impractical. We instead score a deterministic random subset of up to
    max_scan frames - the worst cases in a few-thousand random sample are just
    as informative for eyeballing and it finishes in minutes.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")
    model = load_model_from_ckpt(ckpt, device)

    rows = load_manifest(cfg.manifest, (split,))
    out_dir = Path(cfg.out_dir) / out_subdir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # subsample the rows to score if the split is large
    if len(rows) > max_scan:
        rng0 = random.Random(0)
        scan_indices = sorted(rng0.sample(range(len(rows)), max_scan))
        scan_rows = [rows[i] for i in scan_indices]
        print(f"Split '{split}' has {len(rows)} frames; scoring a random {max_scan}-frame "
              f"subset (use --dump_max_scan to change).")
    else:
        scan_rows = rows
        print(f"Scoring all {len(rows)} {split} frames to find worst cases...")

    dcfg = DataConfig(train_h=cfg.train_h, train_w=cfg.train_w,
                      full_res=cfg.eval_full_res, augment=False,
                      video_cache_size=cfg.video_cache_size)
    ds = RatSegDataset(scan_rows, dcfg, train=False)

    scored = []  # (iou, recall, index-into-scan_rows)
    loader = DataLoader(ds, batch_size=1 if cfg.eval_full_res else cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        prefetch_factor=2 if cfg.num_workers > 0 else None)
    amp_dtype = _amp_dtype(cfg.amp) if device.type == "cuda" else None
    idx = 0
    for img, tgt in loader:
        img = img.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logit = model(img)
        p = (torch.sigmoid(logit.float()) > thresh).float().cpu()
        t = (tgt > 0.5).float()
        for b in range(p.shape[0]):
            inter = (p[b] * t[b]).sum().item()
            union = ((p[b] + t[b]) > 0).float().sum().item()
            tpos = inter
            iou = (inter + 1) / (union + 1)
            rec = (tpos + 1) / (t[b].sum().item() + 1)
            scored.append((iou, rec, idx))
            idx += 1

    # from here on, index into scan_rows (the rows we actually scored)
    rows = scan_rows
    scored.sort(key=lambda s: s[0])  # ascending IoU -> worst first
    worst = scored[:n_worst]
    rng = random.Random(0)
    rest = scored[n_worst:]
    rand = rng.sample(rest, min(n_random, len(rest))) if rest else []
    picks = worst + rand
    print(f"Dumping {len(picks)} overlays ({len(worst)} worst + {len(rand)} random) to {out_dir}")

    for rank, (iou, rec, ri) in enumerate(picks):
        row = rows[ri]
        # decode full-res frame + gt for a clean full-res visualisation
        vr = ds._vr_cache.get(row["video_path"])
        frame_rgb = vr[int(row["frame_idx"])].asnumpy()
        bg_bgr = ds._bg(row["background_path"])
        gt = (np.array(Image.open(row["mask_path"]))
              if False else np.array(Image.open(row["mask_path"])))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt >= 1).astype(np.uint8)

        pred = predict_union_mask(model, frame_rgb, bg_bgr, str(device), thresh=thresh)

        base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gt_ov = base.copy();  gt_ov[gt > 0] = (0, 200, 0)          # green
        pr_ov = base.copy();  pr_ov[pred > 0] = (0, 0, 220)        # red
        gt_ov = cv2.addWeighted(base, 0.5, gt_ov, 0.5, 0)
        pr_ov = cv2.addWeighted(base, 0.5, pr_ov, 0.5, 0)

        tag = "WORST" if rank < len(worst) else "rand"
        label = f"{tag} iou={iou:.3f} rec={rec:.3f} {row['experiment']} cam{row['cam']} f{row['frame_idx']}"
        strip = np.concatenate([base, gt_ov, pr_ov], axis=1)
        cv2.putText(strip, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        fname = f"{rank:03d}_{tag}_iou{iou:.3f}_{row['experiment']}_cam{row['cam']}_f{int(row['frame_idx']):06d}.png"
        cv2.imwrite(str(out_dir / fname), strip)

    print(f"Done. Open {out_dir} - left=frame, middle=ground truth (green), right=prediction (red).")
    print("Missing tail = green with no red; over-segmentation = red with no green.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="seg_run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--video_cache_size", type=int, default=3,
                    help="open decord readers per worker; lower if you hit RAM OOM")
    ap.add_argument("--persistent_workers", action="store_true",
                    help="keep workers (and their video caches) alive across epochs; "
                         "off by default to reclaim RAM each epoch")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--train_h", type=int, default=640)
    ap.add_argument("--train_w", type=int, default=1024)
    ap.add_argument("--eval_full_res", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--test_after", action="store_true",
                    help="after training, evaluate best.pt on the test split per experiment")
    # --- eval-only modes (operate on an existing checkpoint, no training) ---
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to load for --test_only / --dump_overlays "
                         "(default: <out_dir>/best.pt)")
    ap.add_argument("--test_only", action="store_true",
                    help="load --ckpt, run per-experiment test eval, exit (no training)")
    ap.add_argument("--dump_overlays", type=int, default=0, metavar="N",
                    help="load --ckpt and write N worst-IoU + some random overlay PNGs, exit")
    ap.add_argument("--dump_split", default="val",
                    help="which split to dump overlays from (val/test/train)")
    ap.add_argument("--dump_random", type=int, default=20,
                    help="how many random (non-worst) overlays to also dump")
    ap.add_argument("--dump_max_scan", type=int, default=6000,
                    help="max frames to score when finding worst cases (subsampled "
                         "if the split is larger); keeps huge test-set dumps fast")
    ap.add_argument("--sweep_threshold", action="store_true",
                    help="load --ckpt, sweep binarisation thresholds on --dump_split, exit")
    ap.add_argument("--sweep_values", type=float, nargs="*", default=None,
                    help="explicit threshold list for the sweep (default 0.3..0.8)")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = TrainConfig(
        manifest=a.manifest, out_dir=a.out_dir, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, num_workers=a.num_workers,
        grad_accum=a.grad_accum, device=a.device, amp=a.amp,
        train_h=a.train_h, train_w=a.train_w, eval_full_res=a.eval_full_res,
        pretrained=not a.no_pretrained, pos_weight=a.pos_weight,
        video_cache_size=a.video_cache_size, persistent_workers=a.persistent_workers,
    )

    ckpt = a.ckpt or str(Path(cfg.out_dir) / "best.pt")

    # eval-only modes short-circuit training entirely. Force single-process
    # loading (num_workers=0) and a tiny video cache so they cannot trip the
    # RAM OOM killer on the large test videos - eval isn't throughput-critical.
    if a.test_only or a.dump_overlays > 0 or a.sweep_threshold:
        cfg.num_workers = 0
        cfg.video_cache_size = 1

    if a.sweep_threshold:
        print(f"=== THRESHOLD SWEEP on {ckpt} (split '{a.dump_split}') ===")
        sweep_threshold(cfg, ckpt, split=a.dump_split,
                        thresholds=a.sweep_values, max_scan=a.dump_max_scan)
        return

    if a.test_only:
        print(f"=== TEST ONLY on {ckpt} ===")
        run_test(cfg, ckpt, test_split="test")
        return
    if a.dump_overlays > 0:
        print(f"=== DUMP OVERLAYS ({a.dump_overlays} worst + {a.dump_random} random) "
              f"from split '{a.dump_split}' on {ckpt} ===")
        dump_overlays(cfg, ckpt, split=a.dump_split,
                      n_worst=a.dump_overlays, n_random=a.dump_random,
                      max_scan=a.dump_max_scan)
        return

    best = train(cfg)
    if a.test_after:
        print("\n=== TEST (per experiment) ===")
        run_test(cfg, best, test_split="test")


if __name__ == "__main__":
    main()