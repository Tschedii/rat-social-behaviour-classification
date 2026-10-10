from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# -----------------------------------------------------------
# Basic mask utility functions
# -----------------------------------------------------------

def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert any numeric mask to a boolean mask.

    Many segmentation models output masks as floats or integers.
    This helper ensures the mask is strictly boolean so logical
    operations behave correctly.
    """
    return np.asarray(mask).astype(bool)


def mask_area(mask: np.ndarray) -> int:
    """
    Count the number of foreground pixels in a binary mask.
    """
    return int(np.count_nonzero(ensure_binary_mask(mask)))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two masks.

    IoU = intersection / union

    Returns:
        float in [0,1]
    """
    a = ensure_binary_mask(mask_a)
    b = ensure_binary_mask(mask_b)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(inter) / float(union)


def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the centroid (x, y) of a binary mask.

    Returns:
        np.array([x, y]) or None if mask is empty
    """
    m = ensure_binary_mask(mask)

    ys, xs = np.nonzero(m)

    if len(xs) == 0:
        return None

    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Euclidean distance between the centroids
    of two masks.

    If one mask is empty, return infinity.
    """
    ca = mask_centroid(mask_a)
    cb = mask_centroid(mask_b)

    if ca is None or cb is None:
        return float("inf")

    return float(np.linalg.norm(ca - cb))


def area_change_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute relative area change between two masks:

        abs(area_b - area_a) / area_a

    Returns:
        float in [0, inf)

    Notes
    -----
    - If area_a == 0, returns inf unless both are zero, in which case 0.
    """
    area_a = mask_area(mask_a)
    area_b = mask_area(mask_b)

    if area_a == 0:
        return 0.0 if area_b == 0 else float("inf")

    return abs(area_b - area_a) / float(area_a)


# -----------------------------------------------------------
# Decision object describing the result of mask evaluation
# -----------------------------------------------------------

@dataclass
class AbsentDecision:
    """
    Result of evaluating a predicted mask.

    Attributes
    ----------
    present : bool
        Whether the object is considered present in this frame.

    reason : str
        Explanation for the decision.

    area : int
        Pixel area of the mask.

    iou_with_prev : float
        IoU with the previous mask.

    centroid_jump : float
        Distance between centroids of current and previous mask.

    area_change_ratio : float
        Relative change in area vs previous accepted mask.

    reidentified : bool
        True if object was recovered via re-identification.

    became_absent : bool
        True if the object transitioned from present → absent.

    reappeared_while_absent : bool
        True if a mask appeared while object was already absent.

    needs_user_relabel : bool
        Indicates the pipeline should request manual relabel.
    """

    present: bool
    reason: str
    area: int
    iou_with_prev: float
    centroid_jump: float
    area_change_ratio: float = 0.0
    reidentified: bool = False

    # Pipeline control flags
    became_absent: bool = False
    reappeared_while_absent: bool = False
    needs_user_relabel: bool = False


# -----------------------------------------------------------
# Per-object tracking state
# -----------------------------------------------------------

@dataclass
class ObjectTrackState:
    """
    Maintains tracking state for a single object.

    Attributes
    ----------
    obj_id : int
        Unique object identifier.

    is_absent : bool
        Whether the object is currently considered absent.

    absent_count : int
        Number of consecutive frames with bad predictions.

    prev_mask : Optional[np.ndarray]
        Mask accepted in the previous frame.

    last_good_mask : Optional[np.ndarray]
        Last reliable mask used for re-identification.

    waiting_for_manual_reinit : bool
        True when tracker detected reappearance but requires
        manual user relabel before resuming tracking.
    """

    obj_id: int
    is_absent: bool = False
    absent_count: int = 0
    prev_mask: Optional[np.ndarray] = None
    last_good_mask: Optional[np.ndarray] = None

    # Helper state for manual intervention
    waiting_for_manual_reinit: bool = False


# -----------------------------------------------------------
# Configuration controlling absent-object handling
# -----------------------------------------------------------

@dataclass
class AbsentHandlingConfig:
    """
    Configuration parameters for absent object detection.

    These thresholds control how strict the tracker is when
    deciding whether a mask is valid.
    """

    # Minimum allowed mask size (pixels)
    min_area: int = 100

    # Maximum mask size relative to image
    max_area_ratio: float = 0.60

    # Maximum allowed relative area change vs previous accepted mask.
    # Example: 0.20 means the area may not increase or decrease
    # by more than 20% from one accepted frame to the next.
    max_area_change_ratio: float = 0.10

    # A large area change is only treated as a failure if IoU with the previous
    # mask is ALSO below this (i.e. the mask moved off the object, not just the
    # same rat changing pose). Set to 0.0 to disable the guard and reject on
    # area change alone (old strict behaviour). ~0.5 = "if it still overlaps the
    # previous mask by half, a big area change is probably just motion".
    area_change_iou_guard: float = 0.5

    # Minimum IoU with previous mask
    min_iou_with_prev: float = 0.02

    # Maximum allowed centroid movement
    max_centroid_jump: float = 120.0

    # Frames allowed before declaring object absent
    max_bad_frames_before_absent: int = 2

    # Require manual relabel when object reappears
    require_manual_relabel_on_reappearance: bool = True

    # Brief self-recovering dropout tolerance. If an object was absent for no
    # more than `max_brief_absence_frames` and its returning mask overlaps the
    # last-good mask by at least `reappear_auto_accept_min_iou` with a centroid
    # jump no larger than `reappear_auto_accept_max_jump`, the reappearance is
    # auto-accepted instead of flagged for manual relabel. This kills the common
    # false positive where SAM3 blinks for a frame or two and recovers on its
    # own with a good, consistent mask. Sustained absences or spatially
    # inconsistent returns still require review. Set max_brief_absence_frames=0
    # to restore the old always-relabel-on-reappearance behaviour.
    max_brief_absence_frames: int = 3
    reappear_auto_accept_min_iou: float = 0.5
    reappear_auto_accept_max_jump: float = 80.0

    # Optional automatic re-identification
    enable_reid: bool = True
    reid_min_area: int = 100
    reid_min_iou_with_last_good: float = 0.05
    reid_max_centroid_jump: float = 160.0

    # Logging
    verbose: bool = True


# -----------------------------------------------------------
# Main absent-object handling logic
# -----------------------------------------------------------

@dataclass
class AbsentHandler:
    """
    Core logic for determining whether tracked objects
    are present, absent, or reappearing.
    """

    config: AbsentHandlingConfig = field(default_factory=AbsentHandlingConfig)

    def _log(self, msg: str) -> None:
        """Print log messages when verbose mode is enabled."""
        if self.config.verbose:
            print(msg)

    # -------------------------------------------------------
    # Evaluate whether a predicted mask is plausible
    # -------------------------------------------------------

    def evaluate_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, ...],
        prev_mask: Optional[np.ndarray],
        segment_boundary: bool = False,
    ) -> AbsentDecision:
        """
        Evaluate whether a predicted mask is reasonable
        given the previous frame.

        segment_boundary: True on the first frame of a new segment, where SAM3
        has just been re-prompted/re-seeded. The current mask is NOT temporally
        continuous with the previous segment's last frame (different seed), so
        the area-change / centroid-jump / IoU continuity checks are invalid
        here and would spuriously flag every segment transition. On a boundary
        we accept the mask and let it reseed prev_mask, exactly like the very
        first frame. Size sanity (too small / too large) is still enforced.
        """

        h, w = image_shape[:2]
        img_area = h * w

        mask_bool = ensure_binary_mask(mask)
        area = mask_area(mask_bool)
        area_ratio = area / max(img_area, 1)

        # Reject masks that are too small
        if area < self.config.min_area:
            return AbsentDecision(
                present=False,
                reason=f"mask too small ({area} < {self.config.min_area})",
                area=area,
                iou_with_prev=0.0,
                centroid_jump=float("inf"),
                area_change_ratio=0.0,
            )

        # Reject masks that are too large
        if area_ratio > self.config.max_area_ratio:
            return AbsentDecision(
                present=False,
                reason=(
                    f"mask too large "
                    f"({area_ratio:.3f} > {self.config.max_area_ratio:.3f})"
                ),
                area=area,
                iou_with_prev=0.0,
                centroid_jump=float("inf"),
                area_change_ratio=0.0,
            )

        # If no previous mask exists, accept
        if prev_mask is None:
            return AbsentDecision(
                present=True,
                reason="no previous mask; accepted",
                area=area,
                iou_with_prev=1.0,
                centroid_jump=0.0,
                area_change_ratio=0.0,
            )

        # Segment boundary: SAM3 was just re-seeded, so the mask is not
        # temporally continuous with the previous segment. Skip continuity
        # checks (they'd flag every transition) but keep size sanity above.
        if segment_boundary:
            return AbsentDecision(
                present=True,
                reason="segment boundary; continuity check skipped",
                area=area,
                iou_with_prev=1.0,
                centroid_jump=0.0,
                area_change_ratio=0.0,
            )

        prev_mask_bool = ensure_binary_mask(prev_mask)

        # Compute both signals up front so the area check can be qualified by
        # spatial overlap.
        iou_prev = mask_iou(mask_bool, prev_mask_bool)
        jump = centroid_distance(mask_bool, prev_mask_bool)

        # Combined area-change gate: a large area change ALONE is not enough to
        # reject - a rat rearing, turning, or stretching changes its projected
        # area substantially while staying in the same place (high IoU with the
        # previous frame). We only treat a big area change as a real failure if
        # the mask ALSO barely overlaps the previous one (low IoU), which is the
        # signature of the mask jumping to a different object / grabbing
        # background rather than the same rat moving. This keeps the check
        # sensitive to genuine tracking failures while not flagging normal
        # motion (the dominant source of spurious reviews on this footage).
        area_delta_ratio = area_change_ratio(prev_mask_bool, mask_bool)
        if (
            area_delta_ratio > self.config.max_area_change_ratio
            and iou_prev < self.config.area_change_iou_guard
        ):
            prev_area = mask_area(prev_mask_bool)
            return AbsentDecision(
                present=False,
                reason=(
                    f"sudden area change with low overlap: prev_area={prev_area}, "
                    f"curr_area={area}, change={area_delta_ratio:.3f} > "
                    f"{self.config.max_area_change_ratio:.3f}, iou={iou_prev:.3f} < "
                    f"{self.config.area_change_iou_guard:.3f}"
                ),
                area=area,
                iou_with_prev=iou_prev,
                centroid_jump=jump,
                area_change_ratio=area_delta_ratio,
            )

        # Reject if both IoU too small AND movement too large
        if (
            iou_prev < self.config.min_iou_with_prev
            and jump > self.config.max_centroid_jump
        ):
            return AbsentDecision(
                present=False,
                reason=(
                    f"implausible jump: "
                    f"iou={iou_prev:.4f} < {self.config.min_iou_with_prev} and "
                    f"centroid_jump={jump:.1f} > {self.config.max_centroid_jump}"
                ),
                area=area,
                iou_with_prev=iou_prev,
                centroid_jump=jump,
                area_change_ratio=area_delta_ratio,
            )

        # Otherwise accept mask
        return AbsentDecision(
            present=True,
            reason="mask accepted",
            area=area,
            iou_with_prev=iou_prev,
            centroid_jump=jump,
            area_change_ratio=area_delta_ratio,
        )

    # -------------------------------------------------------
    # Attempt automatic re-identification
    # -------------------------------------------------------

    def try_reidentify(
        self,
        state: ObjectTrackState,
        candidate_mask: np.ndarray,
    ) -> Optional[AbsentDecision]:
        """
        Attempt to recover the object using the last known
        reliable mask (geometric re-identification).
        """

        if not self.config.enable_reid:
            return None

        if state.last_good_mask is None:
            return None

        candidate = ensure_binary_mask(candidate_mask)
        area = mask_area(candidate)

        if area < self.config.reid_min_area:
            return None

        iou_last = mask_iou(candidate, state.last_good_mask)
        jump_last = centroid_distance(candidate, state.last_good_mask)
        area_delta_ratio = area_change_ratio(state.last_good_mask, candidate)

        # Optional safety check during re-identification too:
        # do not accept a candidate if its area changed too violently
        # relative to the last good mask.
        if area_delta_ratio > self.config.max_area_change_ratio:
            return None

        # Accept if geometry matches sufficiently
        if (
            iou_last >= self.config.reid_min_iou_with_last_good
            or jump_last <= self.config.reid_max_centroid_jump
        ):
            return AbsentDecision(
                present=True,
                reason=(
                    f"re-identified: iou_with_last_good={iou_last:.4f}, "
                    f"centroid_jump={jump_last:.1f}, "
                    f"area_change={area_delta_ratio:.3f}"
                ),
                area=area,
                iou_with_prev=iou_last,
                centroid_jump=jump_last,
                area_change_ratio=area_delta_ratio,
                reidentified=True,
            )

        return None

    # -------------------------------------------------------
    # Main update function for each frame
    # -------------------------------------------------------

    def update(
        self,
        state: ObjectTrackState,
        predicted_mask: Optional[np.ndarray],
        image_shape: tuple[int, ...],
        frame_idx: Optional[int] = None,
        reinit_mask: Optional[np.ndarray] = None,
        segment_boundary: bool = False,
    ) -> tuple[Optional[np.ndarray], ObjectTrackState, AbsentDecision]:
        """
        Update tracking state for the object.

        Handles:
        - normal tracking
        - disappearance detection
        - re-identification
        - manual relabel requests
        """

        old_absent = state.is_absent

        prefix = (
            f"[frame={frame_idx} obj={state.obj_id}]"
            if frame_idx is not None
            else f"[obj={state.obj_id}]"
        )

        # ---------------------------------------------------
        # Highest priority: external reinitialization
        # ---------------------------------------------------

        if reinit_mask is not None:
            reinit_mask = ensure_binary_mask(reinit_mask)

            state.is_absent = False
            state.absent_count = 0
            state.prev_mask = reinit_mask.copy()
            state.last_good_mask = reinit_mask.copy()
            state.waiting_for_manual_reinit = False

            decision = AbsentDecision(
                present=True,
                reason="reinitialized from external prompt/detector",
                area=mask_area(reinit_mask),
                iou_with_prev=1.0,
                centroid_jump=0.0,
                area_change_ratio=0.0,
                reidentified=True,
            )

            self._log(f"{prefix} REINIT: {decision.reason}")

            return reinit_mask, state, decision

        # ---------------------------------------------------
        # Case: no predicted mask
        # ---------------------------------------------------

        if predicted_mask is None:
            state.absent_count += 1
            became_absent = False

            if state.absent_count >= self.config.max_bad_frames_before_absent:
                if not state.is_absent:
                    became_absent = True
                state.is_absent = True

            decision = AbsentDecision(
                present=False,
                reason="no predicted mask",
                area=0,
                iou_with_prev=0.0,
                centroid_jump=float("inf"),
                area_change_ratio=0.0,
                became_absent=became_absent,
            )

            if became_absent:
                self._log(f"{prefix} ABSENT: object disappeared ({decision.reason})")

            return None, state, decision

        predicted_mask = ensure_binary_mask(predicted_mask)

        # ---------------------------------------------------
        # Case: object already absent but mask appears again
        # ---------------------------------------------------

        if state.is_absent and self.config.require_manual_relabel_on_reappearance:

            area = mask_area(predicted_mask)

            if state.last_good_mask is not None:
                iou_last = mask_iou(predicted_mask, state.last_good_mask)
                jump_last = centroid_distance(predicted_mask, state.last_good_mask)
                area_delta_ratio = area_change_ratio(state.last_good_mask, predicted_mask)
            else:
                iou_last = 0.0
                jump_last = float("inf")
                area_delta_ratio = 0.0

            # Brief self-recovering dropout: if the object was only absent for a
            # short time AND it reappears looking like where it left (good
            # overlap with last-good mask, small centroid jump), SAM3 simply
            # blinked for a frame or two and recovered on its own. Demanding a
            # manual relabel here is a false positive - the returning mask is
            # good and consistent. Auto-accept it. We still demand relabel for
            # SUSTAINED absences (object genuinely lost for many frames) or when
            # the returning mask is spatially inconsistent with last-good (we
            # can't be confident it's the same object / same place).
            brief = state.absent_count <= self.config.max_brief_absence_frames
            consistent = (
                state.last_good_mask is not None
                and iou_last >= self.config.reappear_auto_accept_min_iou
                and jump_last <= self.config.reappear_auto_accept_max_jump
            )
            if brief and consistent:
                _absent_dur = state.absent_count
                self._log(
                    f"{prefix} REAPPEARED-AUTOACCEPT: absent {_absent_dur} "
                    f"frame(s), returning mask consistent "
                    f"(iou={iou_last:.3f}, jump={jump_last:.1f}); accepted without review."
                )
                state.is_absent = False
                state.absent_count = 0
                state.prev_mask = predicted_mask.copy()
                state.last_good_mask = predicted_mask.copy()
                state.waiting_for_manual_reinit = False
                return predicted_mask, state, AbsentDecision(
                    present=True,
                    reason=(
                        f"brief dropout ({_absent_dur} frames) recovered with "
                        f"consistent mask; auto-accepted"
                    ),
                    area=area,
                    iou_with_prev=iou_last,
                    centroid_jump=jump_last,
                    area_change_ratio=area_delta_ratio,
                )

            state.waiting_for_manual_reinit = True

            decision = AbsentDecision(
                present=False,
                reason=(
                    "candidate mask detected for previously absent object; "
                    "manual relabel required"
                ),
                area=area,
                iou_with_prev=iou_last,
                centroid_jump=jump_last,
                area_change_ratio=area_delta_ratio,
                reappeared_while_absent=True,
                needs_user_relabel=True,
            )

            self._log(
                f"{prefix} REAPPEARED: candidate detected for absent object "
                f"(absent {state.absent_count} frames, iou={iou_last:.3f}, "
                f"jump={jump_last:.1f}). Please review / relabel."
            )

            return None, state, decision

        # ---------------------------------------------------
        # Normal evaluation
        # ---------------------------------------------------

        decision = self.evaluate_mask(
            mask=predicted_mask,
            image_shape=image_shape,
            prev_mask=state.prev_mask,
            segment_boundary=segment_boundary,
        )

        if decision.present:
            state.is_absent = False
            state.absent_count = 0
            state.prev_mask = predicted_mask.copy()
            state.last_good_mask = predicted_mask.copy()
            state.waiting_for_manual_reinit = False

            if old_absent and not state.is_absent:
                self._log(f"{prefix} REID: recovered by normal tracking")
            else:
                self._log(
                    f"{prefix} ACCEPT: {decision.reason}; "
                    f"area={decision.area}, "
                    f"iou_prev={decision.iou_with_prev:.4f}, "
                    f"jump={decision.centroid_jump:.1f}, "
                    f"area_change={decision.area_change_ratio:.3f}"
                )

            return predicted_mask, state, decision

        # ---------------------------------------------------
        # Attempt optional re-identification
        # ---------------------------------------------------

        reid_decision = self.try_reidentify(state, predicted_mask)

        if reid_decision is not None:
            state.is_absent = False
            state.absent_count = 0
            state.prev_mask = predicted_mask.copy()
            state.last_good_mask = predicted_mask.copy()
            state.waiting_for_manual_reinit = False

            self._log(f"{prefix} REID: {reid_decision.reason}")

            return predicted_mask, state, reid_decision

        # ---------------------------------------------------
        # Prediction rejected → increase absent counter
        # ---------------------------------------------------

        self._log(
            f"{prefix} REJECT: {decision.reason}; "
            f"area={decision.area}, "
            f"iou_prev={decision.iou_with_prev:.4f}, "
            f"jump={decision.centroid_jump:.1f}, "
            f"area_change={decision.area_change_ratio:.3f}"
        )

        state.absent_count += 1
        became_absent = False

        if state.absent_count >= self.config.max_bad_frames_before_absent:
            if not state.is_absent:
                became_absent = True
            state.is_absent = True

        if became_absent:
            self._log(f"{prefix} ABSENT: object disappeared ({decision.reason})")

        decision.became_absent = became_absent

        # Important: do NOT update prev_mask with bad masks
        return None, state, decision