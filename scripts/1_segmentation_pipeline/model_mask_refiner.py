#!/usr/bin/env python3
"""
model_mask_refiner.py

Fuses the promptless rat-segmenter (best.pt) into the SAM3 pipeline WITHOUT
touching SAM3's identity/propagation. SAM3 decides which pixels belong to
which rat (identity, which it does near-perfectly); this model decides exactly
where the rat pixels are (quality - it recovers the tails SAM3 drops).

Fusion rule (default: "dilated_replace")
----------------------------------------
For each SAM3 object mask M_sam (a bool HxW blob for one rat):
  1. Run the model once on the frame -> a single class-agnostic union
     probability, thresholded to U (all rat pixels, both rats, no identity).
  2. Build a trust region = dilate(M_sam, radius). Only union pixels inside
     this region can be adopted for THIS object - this stops the model's
     occasional far-away shadow blob from being imported, while still letting
     it recover a tail attached to the body.
  3. Refined object mask = (U AND trust_region), assigned to this SAM3 obj_id.
  4. Contested pixels (claimed by >1 SAM3 object, e.g. where two rats touch)
     go to the object whose ORIGINAL SAM3 mask is nearest - i.e. we defer to
     SAM3 exactly in the contact zone where the model's union can't split.

If a refined mask would come out empty (model found nothing near SAM3's blob),
we KEEP the original SAM3 mask for that object - refinement must never delete
an object SAM3 was tracking. This guarantees "same or a tiny bit better", never
worse on identity/recall.

The union is computed ONCE per frame and shared across objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import cv2


class ModelMaskRefiner:
    def __init__(
        self,
        ckpt_path: str,
        background_bgr: np.ndarray,
        device: str = "cuda:0",
        train_h: int = 640,
        train_w: int = 1024,
        thresh: float = 0.5,
        dilate_radius: int = 25,
        mode: str = "dilated_replace",   # dilated_replace | replace | add_only | off
        min_refined_pixels: int = 20,
        min_iou_vs_sam: float = 0.5,
    ):
        import os as _os
        import sys as _sys
        import torch

        # train_segmenter.py and prepare_dataset.py live alongside best.pt
        # (the sam3_rat_training dir). The pipeline is launched from a sibling
        # directory, so that dir may not be on the import path - add it.
        _ckpt_dir = _os.path.dirname(_os.path.abspath(ckpt_path))
        if _ckpt_dir and _ckpt_dir not in _sys.path:
            _sys.path.insert(0, _ckpt_dir)

        try:
            from train_segmenter import ResNet34UNet, IMAGENET_MEAN, IMAGENET_STD
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import train_segmenter from {_ckpt_dir}. "
                f"Ensure train_segmenter.py sits in the same directory as best.pt. "
                f"(original: {e})"
            )
        try:
            from prepare_dataset import compute_foreground_prior
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import prepare_dataset from {_ckpt_dir}. "
                f"Ensure prepare_dataset.py sits in the same directory as best.pt. "
                f"(original: {e})"
            )

        self._torch = torch
        self._compute_prior = compute_foreground_prior
        self._mean = IMAGENET_MEAN
        self._std = IMAGENET_STD

        self.device = device if torch.cuda.is_available() or "cpu" in device else "cpu"
        self.train_h = train_h
        self.train_w = train_w
        self.thresh = thresh
        self.dilate_radius = dilate_radius
        self.mode = mode
        self.min_refined_pixels = min_refined_pixels
        self.min_iou_vs_sam = min_iou_vs_sam
        self.background_bgr = background_bgr

        self.model = ResNet34UNet(in_ch=4, pretrained=False).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
        )
        print(f"[refiner] loaded {ckpt_path} (mode={mode}, thresh={thresh}, "
              f"dilate={dilate_radius}px, device={self.device})")

    # -- model forward: full-res union probability for one RGB frame --
    def _union_prob(self, frame_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        H0, W0 = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        bg = self.background_bgr
        if bg.shape[:2] != (H0, W0):
            bg = cv2.resize(bg, (W0, H0), interpolation=cv2.INTER_AREA)
        prior_full = self._compute_prior(frame_bgr, bg)  # HxW float32 [0,1]

        rgb = cv2.resize(frame_rgb, (self.train_w, self.train_h),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        pr = cv2.resize(prior_full, (self.train_w, self.train_h),
                        interpolation=cv2.INTER_AREA)
        x = np.concatenate([rgb, pr[..., None]], axis=2).transpose(2, 0, 1)[None]
        xt = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            logit = self.model(xt)
            prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()

        # upsample probability to full res (bilinear keeps thin tails smooth)
        return cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)

    def refine_candidates(
        self,
        frame_rgb: np.ndarray,
        candidates: Dict[int, np.ndarray],
        obj_ids_sorted: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        candidates: {obj_id: HxW bool} from SAM3. Returns the SAME structure
        with refined pixels, identity preserved.
        """
        if self.mode == "off" or not candidates:
            return candidates

        prob = self._union_prob(frame_rgb)
        union = prob >= self.thresh   # HxW bool, both rats, no identity

        if not union.any():
            # model sees no rat -> trust SAM3 entirely for this frame
            return candidates

        refined: Dict[int, np.ndarray] = {}
        # precompute per-object SAM3 masks and their trust regions
        sam_masks = {oid: np.asarray(m).astype(bool) for oid, m in candidates.items()}

        if self.mode == "add_only":
            # union pixels are ADDED to each SAM3 mask within its trust region,
            # never removed; contested handled by nearest below
            trust = {oid: cv2.dilate(m.astype(np.uint8), self._dilate_kernel).astype(bool)
                     for oid, m in sam_masks.items()}
            claim = {oid: sam_masks[oid] | (union & trust[oid]) for oid in sam_masks}
        elif self.mode == "replace":
            # union pixels replace SAM3 entirely, no trust gating; contested by nearest
            claim = {oid: union.copy() for oid in sam_masks}
        else:  # dilated_replace (default)
            trust = {oid: cv2.dilate(m.astype(np.uint8), self._dilate_kernel).astype(bool)
                     for oid, m in sam_masks.items()}
            claim = {oid: union & trust[oid] for oid in sam_masks}

        # resolve contested pixels (claimed by >1 object) by nearest ORIGINAL
        # SAM3 mask - defer to SAM3 in the contact zone
        oids = list(claim.keys())
        if len(oids) >= 2:
            stack = np.stack([claim[o] for o in oids], axis=0)  # K,H,W
            claim_count = stack.sum(axis=0)                      # H,W
            contested = claim_count > 1
            if contested.any():
                # distance to each object's original SAM3 mask
                dists = []
                for o in oids:
                    m = sam_masks[o].astype(np.uint8)
                    if m.any():
                        d = cv2.distanceTransform((1 - m).astype(np.uint8),
                                                  cv2.DIST_L2, 3)
                    else:
                        d = np.full(union.shape, 1e6, dtype=np.float32)
                    dists.append(d)
                dstack = np.stack(dists, axis=0)          # K,H,W
                nearest = np.argmin(dstack, axis=0)        # H,W -> index into oids
                for ci, o in enumerate(oids):
                    # in contested pixels, keep only where this obj is nearest
                    keep = claim[o] & (~contested | (nearest == ci))
                    claim[o] = keep

        # finalise. Two safeguards, both ensuring the refiner can only SHARPEN
        # SAM3's mask, never hand the tracker something structurally different:
        #   (1) too few pixels -> fall back to SAM3 (never delete an object)
        #   (2) refined mask diverges too far from SAM3's OWN mask for this
        #       object (IoU < max_iou_change floor) -> fall back to SAM3. This
        #       prevents the pathology where the model under-segments one rat,
        #       producing a small/offset refined mask that has ~0 IoU with the
        #       previous frame and trips the tracker's area-change gate. The
        #       refiner's job is pixel-level improvement, so a large structural
        #       change means the fusion went wrong and must be discarded.
        for oid in obj_ids_sorted if obj_ids_sorted else oids:
            if oid not in claim:
                if oid in sam_masks:
                    refined[oid] = sam_masks[oid]
                continue
            m = claim[oid].astype(bool)
            sam_m = sam_masks[oid]

            if int(m.sum()) < self.min_refined_pixels:
                refined[oid] = sam_m               # (1) fall back to SAM3
                continue

            # (2) IoU between refined and SAM3's own mask for this object
            inter = int(np.logical_and(m, sam_m).sum())
            union_a = int(np.logical_or(m, sam_m).sum())
            iou_vs_sam = inter / union_a if union_a > 0 else 0.0
            if iou_vs_sam < self.min_iou_vs_sam:
                refined[oid] = sam_m               # refinement diverged; discard
            else:
                refined[oid] = m

        # any object SAM3 had but we somehow dropped -> keep original
        for oid, m in sam_masks.items():
            refined.setdefault(oid, m)

        return refined


def compute_video_background_bgr(video_path: str, max_samples: int = 150,
                                 band_rows: int = 300) -> np.ndarray:
    """Per-pixel temporal median over a strided sample of the FULL video.
       Identical approach to prepare_dataset, returns BGR uint8."""
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    idx = np.linspace(0, n - 1, min(max_samples, n)).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    frames_rgb = vr.get_batch(idx).asnumpy()   # S,H,W,3 RGB
    S, H, W, _ = frames_rgb.shape
    bg_rgb = np.empty((H, W, 3), dtype=np.uint8)
    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        bg_rgb[y0:y1] = np.median(frames_rgb[:, y0:y1], axis=0).astype(np.uint8)
    return cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
