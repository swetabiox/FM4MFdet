# scripts/infer_wsi.py

"""
Whole-slide sliding-window inference for mitotic figure detection,
scored in the style of the official MIDOG 2025 Challenge evaluation.

WHAT THIS DOES
--------------
The detectors in this repository are trained and tested on small patches
(1008x1008 or 1024x1024). This script instead evaluates a trained model on the
*whole* MIDOG++ region-of-interest images (~7215x5412 px), the way the model
would be used in practice:

  1. Slide a fixed-size window over each WSI/ROI with overlap.
  2. Run the detector on every window.
  3. Project every detection from window-local coordinates back into
     whole-slide coordinates by adding the window origin.
  4. Merge detections across overlapping window seams with Non-Maximum
     Suppression, so a mitosis seen in two adjacent windows is counted once.
  5. Score the merged slide-level detections against the MIDOG++ annotations
     using the MIDOG evaluation criterion (see below).

THRESHOLD OPTIMISATION (val -> test)
------------------------------------
The detection score threshold (`det_thresh`) is a free parameter and must be
chosen on data the test set never sees. This script therefore runs in two
phases:

  PHASE 1 - inference (threshold-free):
    Sliding-window inference is run ONCE per slide and ALL detections above a
    low floor (`--score-floor`, default 0.0) are kept. NMS for seam-merging is
    applied at this stage (it is independent of the final det_thresh). The raw
    per-slide detections are cached to disk so phase 2 never re-runs the model.

  PHASE 2 - scoring (cheap, threshold-dependent):
    The cached detections are scored at many candidate thresholds. The
    threshold that maximises pooled (micro) MIDOG F1 on the VALIDATION slides
    is selected, then the TEST slides are scored once at that fixed threshold.
    Because scoring is just KD-tree matching on cached points, sweeping
    hundreds of thresholds is essentially free.

This removes the manual "tune --score-thr on val, then re-run on test" loop
that the old single-split script required.

GEOMETRY CONSISTENCY (window == Resize == backbone img_size)
------------------------------------------------------------
The boxes the model returns are in the frame of the image it was actually fed,
which is the size produced by the pipeline's `Resize` step and (for ViT
backbones) further forced to the backbone's `img_size`. Projecting those boxes
back to whole-slide coordinates with `box += window_origin` is ONLY correct
when the window side, the `Resize` scale, and the backbone `img_size` all
agree. If they diverge (e.g. running a 1008 window through a UNI config whose
backbone forces 1024), the returned boxes are in a different scale than the
window slice and the projection is silently wrong. This script asserts the
three agree before running; for UNI configs pass `--window 1024`.

MIDOG-STYLE SCORING
-------------------
This follows utils/eval_utils.py from the official MIDOG 2025 Guide:

  * Mitoses are point annotations. A predicted box matches a ground-truth
    point when the box CENTRE lies within `radius` pixels of the point.
    Default radius = 25 px (the official value; bbox side is 50 px).
  * Matching is a one-to-one greedy assignment by spatial proximity
    (`evalutils.scorers.score_detection`, a KD-tree radius query) - it is
    NOT ordered by detection score. Detections are first filtered by
    score > det_thresh.
  * F1 is computed the MIDOG way:  F1 = 2*TP / (2*TP + FP + FN).
  * Aggregate precision / recall / F1 are POOLED over all slides (micro).
    A per-tumour-type breakdown is also produced, mirroring MIDOG.

If the `evalutils` package is installed (it is in the MIDOG Guide
requirements) its `score_detection` is used directly, so numbers match the
challenge exactly. Otherwise a faithful built-in KD-tree fallback is used.

This is different from test_mmdet.py, which scores per patch with COCO mAP.

GROUND-TRUTH BBOX FORMAT (corners vs xywh)
------------------------------------------
A COCO `bbox` is conventionally `[x, y, width, height]`, but some MIDOG++
exports store `[x1, y1, x2, y2]` corners. Reading one as the other silently
shifts every ground-truth centre. This script auto-detects the format from the
annotation file (see `detect_bbox_format`) and can be overridden with
`--bbox-format`. The detector must match how the SAME json was read at training
time by `CocoDataset` (which assumes xywh).

CHANNEL ORDER (RGB vs BGR)
--------------------------
At training, MMDetection's `LoadImageFromFile` reads images with OpenCV, which
yields BGR. The model's `data_preprocessor` then applies `bgr_to_rgb=True`,
so the network is trained on RGB pixels. This script loads windows directly
into memory (PIL, already RGB) and removes `LoadImageFromFile` from the
pipeline, so no load-time conversion happens. To feed the model exactly what
it saw in training, the window is converted to the channel order that the
config's `data_preprocessor` expects as INPUT: if `bgr_to_rgb=True` the
preprocessor wants BGR (it will flip to RGB), so the window is handed over as
BGR; if `bgr_to_rgb=False` the window is handed over as RGB. Either way the
network ends up seeing RGB, matching training. Getting this wrong silently
swaps the red and blue channels — which for H&E means swapping the eosin and
hematoxylin signals, degrading detections without any error.

PATIENT SCOPE
-------------
Pass `--slides` (a split manifest) so the script knows which slides are the
validation patients (threshold selection) and which are the test patients
(final reporting). Both splits must be present in the manifest.

LOGGING
-------
Everything printed to the console is also written to a timestamped log file in
`--out-dir` (or to `--log-file` if given), so each run is reproducible from its
log. Use `--quiet` to suppress console output while still writing the file.

USAGE
-----
    python scripts/infer_wsi.py \\
        --config configs/faster_rcnn_uni_midogpp.py \\
        --checkpoint outputs/work_dirs/.../best_coco_bbox_mAP_epoch_12.pth \\
        --roi-dir data/Datensatz/MIDOGpp_ROIs \\
        --ann-file data/MIDOGpp.json \\
        --slides data/coco_annotations/patches_1024/split_manifest.json \\
        --window 1024 \\
        --out-dir outputs/wsi/faster_rcnn_uni_1024

Key options:
    --window       window side in px (default 1008; use 1024 for UNI configs)
    --overlap      window overlap fraction (default 0.30, the MIDOG default)
    --score-floor  low score floor kept during inference (default 0.0). All
                   detections above this are cached; the final det_thresh is
                   chosen on val at or above this floor.
    --thr-min/--thr-max/--thr-step
                   candidate det_thresh grid for the val sweep
                   (default 0.05 .. 0.95 step 0.01).
    --nms-iou      IoU threshold for seam-merging NMS (default 0.30)
    --radius       MIDOG match radius in px (default 25)
    --bbox-format  ground-truth bbox format: auto | xywh | xyxy (default auto)
    --device       inference device (default cuda:0)
    --log-file     explicit log file path (default: <out-dir>/infer_wsi_<ts>.log)
    --quiet        do not echo log lines to the console
    --reuse-detections
                   skip phase 1 and load cached wsi_detections_raw.json from
                   --out-dir (re-tune thresholds without re-running the model).
"""

import argparse
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmdet.apis import init_detector

_orig_torch_load = torch.load
def _torch_load_full(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_full

Image.MAX_IMAGE_PIXELS = None  # ROI images are large


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# A module-level logger replaces bare print(). It writes to a file always and
# to the console unless --quiet. `log()` is a thin wrapper so the existing
# call sites read naturally.

LOGGER = logging.getLogger("infer_wsi")


def setup_logging(log_path: Path, quiet: bool) -> None:
    """Configure LOGGER to write to `log_path` and (optionally) the console."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)

    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        # console stays clean (no timestamp prefix); the file keeps the detail
        sh.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(sh)

    LOGGER.info("Logging to %s", log_path)


def log(msg: str = "") -> None:
    """Drop-in for print() that routes through LOGGER (file + console)."""
    LOGGER.info(msg)


# ---------------------------------------------------------------------------
# Slide id (kept identical to make_patient_splits.py / tile_rois.py)
# ---------------------------------------------------------------------------

_TILE_SUFFIX = re.compile(
    r"(?:[_-](?:x\d+[_-]y\d+|tile[_-]?\d+|patch[_-]?\d+|\d+[_-]\d+|\d+))+$",
    re.IGNORECASE,
)


def slide_id_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stripped = _TILE_SUFFIX.sub("", stem)
    return stripped if stripped else stem


# ---------------------------------------------------------------------------
# Sliding window geometry
# ---------------------------------------------------------------------------

def window_origins(extent: int, window: int, stride: int):
    """Start coordinates of windows along one axis.

    The final window is clamped to the image edge so the last strip is always
    covered even when the image size is not a multiple of the stride. This
    matches the coordinate generation in the MIDOG Guide's inference dataset
    (`min(x, width - size)`).
    """
    if extent <= window:
        return [0]
    origins = list(range(0, extent - window + 1, stride))
    last = extent - window
    if origins[-1] != last:
        origins.append(last)
    return origins


# ---------------------------------------------------------------------------
# Geometry consistency check
# ---------------------------------------------------------------------------

def _resize_scale_from_pipeline(cfg):
    """Return the square side enforced by the test pipeline's Resize, or None.

    Looks for a `Resize` step with a `scale` of (s, s); returns s. If the
    Resize is non-square it returns the tuple so the caller can complain.
    """
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "Resize":
            scale = t.get("scale")
            if isinstance(scale, (tuple, list)) and len(scale) == 2:
                return tuple(int(v) for v in scale)
            if isinstance(scale, int):
                return (scale, scale)
    return None


def _backbone_img_size(cfg):
    """Return backbone.img_size if the config sets one (ViT backbones), else None."""
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        bb = model_cfg.get("backbone", {})
        if isinstance(bb, dict):
            v = bb.get("img_size")
            if v is not None:
                return int(v)
    return None


def assert_geometry_consistent(cfg, window: int) -> None:
    """Abort if window, Resize scale, and backbone img_size disagree.

    Projecting window-local detections to slide coordinates assumes the model
    output frame equals the window slice. That holds only when these three
    sizes coincide. See the GEOMETRY CONSISTENCY note in the module docstring.
    """
    problems = []

    resize = _resize_scale_from_pipeline(cfg)
    if resize is None:
        log("WARNING: no square Resize found in the test pipeline; cannot "
            "verify window/Resize consistency. Make sure the model is fed "
            "exactly `window` px.")
    else:
        rw, rh = resize
        if rw != rh:
            problems.append(f"Resize scale is non-square {resize}; this script "
                            f"assumes a square window.")
        elif rw != window:
            problems.append(
                f"--window={window} != Resize scale={rw}. The model would be "
                f"fed {rw}px while boxes are projected as if {window}px. "
                f"Pass --window {rw}.")

    img_size = _backbone_img_size(cfg)
    if img_size is not None and img_size != window:
        problems.append(
            f"--window={window} != backbone img_size={img_size}. This backbone "
            f"resizes every window to {img_size}px internally, so returned "
            f"boxes are in {img_size}px space, not {window}px. Pass "
            f"--window {img_size}.")

    if problems:
        log("ABORTING: window / Resize / backbone img_size are inconsistent:")
        for p in problems:
            log(f"  - {p}")
        raise SystemExit(2)

    pieces = [f"window={window}"]
    if resize is not None:
        pieces.append(f"Resize={resize[0]}")
    if img_size is not None:
        pieces.append(f"backbone.img_size={img_size}")
    log(f"Geometry OK: {', '.join(pieces)} all agree; box projection is valid.")


# ---------------------------------------------------------------------------
# NMS  (pure numpy, greedy, IoU-based) -- for merging window seams
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS. boxes are [x1,y1,x2,y2]. Returns kept indices."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou < iou_thr]

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# MIDOG-style detection scoring
# ---------------------------------------------------------------------------

def _centers(xyxy: np.ndarray) -> np.ndarray:
    """Box centers from [x1,y1,x2,y2] rows -> (N,2) array of (cx,cy)."""
    if xyxy.size == 0:
        return np.zeros((0, 2), dtype=float)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) * 0.5,
                     (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1)


try:
    # The official MIDOG evaluation uses this exact scorer.
    from evalutils.scorers import score_detection as _evalutils_score
    _HAVE_EVALUTILS = True
except Exception:                                   # pragma: no cover
    _evalutils_score = None
    _HAVE_EVALUTILS = False


def _score_detection_fallback(gt_points: np.ndarray,
                              pred_points: np.ndarray,
                              radius: float):
    """KD-tree radius matching, a faithful re-implementation of
    evalutils.scorers.score_detection.

    One-to-one greedy assignment by spatial proximity: among all
    (prediction, ground-truth) pairs whose centres are within `radius`, the
    closest pairs are committed first; each point is used at most once.
    This is NOT ordered by detection score - score filtering happens before
    this function is called, exactly as in the MIDOG code.

    Returns (tp, fp, fn).
    """
    G, P = len(gt_points), len(pred_points)
    if G == 0 and P == 0:
        return 0, 0, 0
    if G == 0:
        return 0, P, 0
    if P == 0:
        return 0, 0, G

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(gt_points)
        pairs = []
        for pi in range(P):
            for gi in tree.query_ball_point(pred_points[pi], r=radius):
                d = float(np.hypot(*(pred_points[pi] - gt_points[gi])))
                pairs.append((d, pi, gi))
    except Exception:
        # last-resort O(P*G) dense computation
        pairs = []
        for pi in range(P):
            for gi in range(G):
                d = float(np.hypot(*(pred_points[pi] - gt_points[gi])))
                if d <= radius:
                    pairs.append((d, pi, gi))

    pairs.sort(key=lambda t: t[0])
    gt_used = np.zeros(G, dtype=bool)
    pred_used = np.zeros(P, dtype=bool)
    tp = 0
    for _d, pi, gi in pairs:
        if not pred_used[pi] and not gt_used[gi]:
            pred_used[pi] = True
            gt_used[gi] = True
            tp += 1
    return tp, P - tp, G - tp


def _score_points(gt_points: np.ndarray,
                  pred_points: np.ndarray,
                  radius: float):
    """Score one slide's already-thresholded prediction points the MIDOG way.

    Returns (tp, fp, fn).
    """
    if _HAVE_EVALUTILS:
        res = _evalutils_score(
            ground_truth=np.asarray(gt_points, dtype=float).reshape(-1, 2),
            predictions=np.asarray(pred_points, dtype=float).reshape(-1, 2),
            radius=radius,
        )
        return (int(res.true_positives),
                int(res.false_positives),
                int(res.false_negatives))

    return _score_detection_fallback(
        np.asarray(gt_points, dtype=float).reshape(-1, 2),
        np.asarray(pred_points, dtype=float).reshape(-1, 2),
        radius)


def score_slide(gt_points: np.ndarray,
                det_xyxy: np.ndarray,
                det_scores: np.ndarray,
                det_thresh: float,
                radius: float):
    """Score one slide the MIDOG way, filtering detections at det_thresh.

    gt_points  : (G,2) ground-truth mitosis centres.
    det_xyxy   : (D,4) predicted boxes in whole-slide coords.
    det_scores : (D,)  predicted scores.
    det_thresh : keep detections with score > det_thresh (strict, as in MIDOG).
    radius     : match radius in px.

    Returns (tp, fp, fn). F1 is derived by the caller with midog_f1().
    """
    keep = det_scores > det_thresh
    pred_points = _centers(det_xyxy[keep])
    return _score_points(gt_points, pred_points, radius)


def midog_f1(tp: int, fp: int, fn: int) -> float:
    """MIDOG F1:  2*TP / (2*TP + FP + FN)."""
    eps = 1e-12
    return (2.0 * tp) / ((2.0 * tp) + fp + fn + eps)


def precision_recall(tp: int, fp: int, fn: int):
    eps = 1e-12
    return tp / (tp + fp + eps), tp / (tp + fn + eps)


# ---------------------------------------------------------------------------
# FROC  (Free-response ROC: sensitivity vs mean false positives per slide)
# ---------------------------------------------------------------------------
#
# FROC is the standard detection metric for the MIDOG family. Unlike F1 (which
# fixes one operating threshold) it sweeps the detection threshold and traces
# sensitivity (recall) against the average number of false positives per scored
# slide. The single-number summary ("FROC score") averages sensitivity at a
# fixed set of FP/slide operating points -- here the de-facto MIDOG set
# {0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8} fppi.
#
# This script is single-class (category_id == 1, mitotic figures), so there is
# no per-detection-class FROC to compute. The only breakdown the annotations
# support is per TUMOUR TYPE (a grouping of slides), which is what the
# "individual" curves below are -- they are labelled per-tumour, not per-class,
# to avoid implying a class axis the data does not have.

# Standard FROC operating points (false positives per slide).
FROC_FPPI = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def froc_curve(det_arrays, gt_points, slides, thr_grid, radius):
    """Trace an FROC curve over `slides` by sweeping the detection threshold.

    For every threshold in `thr_grid` the pooled TP/FP/FN over `slides` are
    computed (reusing the same MIDOG matching as F1), then converted to
    sensitivity = TP/(TP+FN) and fppi = FP/n_slides.

    Returns a list of points sorted by ascending fppi, each a dict with
    threshold, sensitivity, fppi, tp, fp, fn. `slides` that are missing from
    det_arrays/gt_points are still counted in n_slides (they contribute FN/empty),
    matching how the split is scored elsewhere.
    """
    n_slides = max(1, len(slides))
    pts = []
    for thr in thr_grid:
        TP = FP = FN = 0
        for slide in slides:
            boxes, scores = det_arrays.get(
                slide, (np.zeros((0, 4)), np.zeros((0,))))
            gt = gt_points.get(slide, np.zeros((0, 2), dtype=float))
            tp, fp, fn = score_slide(gt, boxes, scores, thr, radius)
            TP += tp; FP += fp; FN += fn
        sens = TP / (TP + FN + 1e-12)
        fppi = FP / n_slides
        pts.append(dict(threshold=float(thr), sensitivity=float(sens),
                        fppi=float(fppi), tp=int(TP), fp=int(FP), fn=int(FN)))
    # Sort by fppi ascending (threshold high -> low gives increasing fppi).
    pts.sort(key=lambda d: d["fppi"])
    return pts


def froc_score(curve, fppi_points=FROC_FPPI):
    """Mean sensitivity interpolated at the standard FP/slide operating points.

    `curve` is the output of froc_curve (sorted by ascending fppi). At each
    target fppi we take the highest sensitivity achievable without exceeding
    that fppi (a step/greatest-lower-bound read of the curve, the convention
    used by the MIDOG evaluation). Targets beyond the curve's max fppi take the
    curve's maximum sensitivity; targets below its min fppi contribute 0.

    Returns (froc_score, per_point) where per_point maps each target fppi to the
    sensitivity credited there.
    """
    per_point = {}
    if not curve:
        for f in fppi_points:
            per_point[f] = 0.0
        return 0.0, per_point

    max_fppi = curve[-1]["fppi"]
    max_sens_overall = max(p["sensitivity"] for p in curve)

    for f in fppi_points:
        if f >= max_fppi:
            # at or beyond the densest operating point we measured, the best
            # sensitivity seen on the curve is achievable
            per_point[f] = max_sens_overall
            continue
        # highest sensitivity among operating points with fppi <= target
        cands = [p["sensitivity"] for p in curve if p["fppi"] <= f]
        per_point[f] = max(cands) if cands else 0.0

    score = float(np.mean([per_point[f] for f in fppi_points]))
    return score, per_point


def _froc_block(det_arrays, gt_points, slides, thr_grid, radius):
    """Compute curve + score + per-point for one set of slides.

    Returns a dict ready to drop into the metrics JSON.
    """
    curve = froc_curve(det_arrays, gt_points, slides, thr_grid, radius)
    score, per_point = froc_score(curve)
    return dict(
        froc_score=score,
        fppi_points=list(FROC_FPPI),
        sensitivity_at_fppi={f"{f:g}": per_point[f] for f in FROC_FPPI},
        curve=curve,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect_window(model, pipeline, window_rgb: np.ndarray, bgr_to_rgb: bool):
    """Run the detector on one window (HxWx3 uint8, RGB). Returns (xyxy, scores).

    `window_rgb` is RGB (as loaded by PIL). The model's data_preprocessor will
    apply `bgr_to_rgb` to whatever we pass it, so we feed it the channel order
    it expects as INPUT: BGR when bgr_to_rgb=True (it flips to RGB), RGB when
    bgr_to_rgb=False. The network therefore always sees RGB, exactly as in
    training. See the CHANNEL ORDER note in the module docstring.
    """
    img = window_rgb[:, :, ::-1] if bgr_to_rgb else window_rgb
    img = np.ascontiguousarray(img)
    data = dict(
        img=img,
        img_id=0,
        img_path='<window>',
        ori_shape=img.shape[:2],
        img_shape=img.shape[:2],
    )
    data = pipeline(data)
    result = model.test_step(default_collate([data]))[0]
    inst = result.pred_instances
    if inst is None or len(inst) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return (inst.bboxes.detach().cpu().numpy().astype(np.float32),
            inst.scores.detach().cpu().numpy().astype(np.float32))


def build_inference_pipeline(cfg):
    """Test pipeline with the disk-loading steps removed (pixels are already
    in memory and there are no per-window annotations).

    Also returns the `bgr_to_rgb` flag from the model's data_preprocessor so
    the caller can hand windows to the model in the channel order the
    preprocessor expects as input (see the CHANNEL ORDER note in the module
    docstring). Defaults to True, which is the MMDetection default for
    DetDataPreprocessor.
    """
    steps = []
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get('type') in ('LoadImageFromFile', 'LoadAnnotations'):
            continue
        steps.append(t)

    # The effective preprocessor is the one inside model=dict(...); fall back
    # to a top-level data_preprocessor, then to the MMDetection default.
    dp = {}
    model_cfg = cfg.get('model', {})
    if isinstance(model_cfg, dict):
        dp = model_cfg.get('data_preprocessor') or {}
    if not dp:
        dp = cfg.get('data_preprocessor') or {}
    bgr_to_rgb = dp.get('bgr_to_rgb', True)

    return Compose(steps), bool(bgr_to_rgb)


def infer_slide(model, pipeline, bgr_to_rgb, roi, window, stride,
                nms_iou, score_floor, desc=None):
    """Run sliding-window inference on one ROI array and seam-merge with NMS.

    Keeps every detection with score > score_floor. Returns (xyxy, scores) in
    whole-slide coordinates. A nested tqdm bar tracks windows within the slide
    (this is where inference time is spent); pass `desc` to label it.
    """
    H, W = roi.shape[:2]
    ys = window_origins(H, window, stride)
    xs = window_origins(W, window, stride)
    grid = [(oy, ox) for oy in ys for ox in xs]

    det_boxes, det_scores = [], []
    for oy, ox in tqdm(grid, desc=desc or "windows", unit="win",
                       leave=False):
        win = roi[oy:oy + window, ox:ox + window]
        if win.shape[0] != window or win.shape[1] != window:
            canvas = np.zeros((window, window, 3), dtype=win.dtype)
            canvas[:win.shape[0], :win.shape[1]] = win
            win = canvas

        boxes, scores = detect_window(model, pipeline, win, bgr_to_rgb)
        if boxes.shape[0] == 0:
            continue
        if score_floor > 0.0:
            m = scores > score_floor
            boxes, scores = boxes[m], scores[m]
            if boxes.shape[0] == 0:
                continue
        # project window-local boxes into whole-slide coordinates
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy
        det_boxes.append(boxes)
        det_scores.append(scores)

    if det_boxes:
        det_boxes = np.concatenate(det_boxes, axis=0)
        det_scores = np.concatenate(det_scores, axis=0)
        kept = nms(det_boxes, det_scores, nms_iou)
        return det_boxes[kept], det_scores[kept]
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)


# ---------------------------------------------------------------------------
# Threshold optimisation (validation sweep)
# ---------------------------------------------------------------------------

def pooled_f1_at(det_arrays, gt_points, slides, det_thresh, radius):
    """Pooled (micro) MIDOG F1 over `slides` at a given det_thresh.

    det_arrays : slide -> (xyxy, scores)
    gt_points  : slide -> (G,2)
    Returns (f1, precision, recall, TP, FP, FN).
    """
    TP = FP = FN = 0
    for slide in slides:
        boxes, scores = det_arrays.get(
            slide, (np.zeros((0, 4)), np.zeros((0,))))
        gt = gt_points.get(slide, np.zeros((0, 2), dtype=float))
        tp, fp, fn = score_slide(gt, boxes, scores, det_thresh, radius)
        TP += tp; FP += fp; FN += fn
    p, r = precision_recall(TP, FP, FN)
    return midog_f1(TP, FP, FN), p, r, TP, FP, FN


def optimise_threshold(det_arrays, gt_points, val_slides,
                       thr_grid, radius):
    """Pick the det_thresh maximising pooled val MIDOG F1.

    Returns (best_thr, sweep) where sweep is a list of dicts for every
    candidate threshold (useful for plotting / the report).
    """
    sweep = []
    best = None
    for thr in tqdm(thr_grid, desc="val threshold sweep", unit="thr",
                    leave=False):
        f1, p, r, TP, FP, FN = pooled_f1_at(
            det_arrays, gt_points, val_slides, thr, radius)
        row = dict(threshold=float(thr), f1=f1, precision=p, recall=r,
                   tp=TP, fp=FP, fn=FN)
        sweep.append(row)
        # tie-break: prefer higher F1, then higher threshold (fewer FPs)
        key = (f1, thr)
        if best is None or key > best[0]:
            best = (key, thr)
    return float(best[1]), sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Whole-slide sliding-window inference with MIDOG-style "
                    "evaluation and val->test threshold optimisation.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--roi-dir", required=True,
                   help="Directory of whole-slide ROI images (.tiff).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ann-file", required=True,
                   help="MIDOG++ annotation JSON (required for scoring and "
                        "threshold optimisation).")
    p.add_argument("--slides", required=True,
                   help="split_manifest.json containing slides.val and "
                        "slides.test. The threshold is optimised on val and "
                        "reported on test.")
    p.add_argument("--window", type=int, default=1008)
    p.add_argument("--overlap", type=float, default=0.30,
                   help="Window overlap fraction (MIDOG default 0.30).")
    p.add_argument("--score-floor", type=float, default=0.0,
                   help="Low score floor kept during inference. All "
                        "detections above this are cached; the final "
                        "det_thresh is selected on val at or above this floor.")
    p.add_argument("--thr-min", type=float, default=0.05,
                   help="Smallest candidate det_thresh in the val sweep.")
    p.add_argument("--thr-max", type=float, default=0.95,
                   help="Largest candidate det_thresh in the val sweep.")
    p.add_argument("--thr-step", type=float, default=0.01,
                   help="Step of the candidate det_thresh grid.")
    p.add_argument("--nms-iou", type=float, default=0.30,
                   help="IoU threshold for seam-merging NMS "
                        "(MIDOG default 0.30).")
    p.add_argument("--radius", type=float, default=25.0,
                   help="MIDOG match radius in px (default 25).")
    p.add_argument("--bbox-format", choices=("auto", "xywh", "xyxy"),
                   default="auto",
                   help="Ground-truth bbox format in --ann-file. 'auto' "
                        "detects it; override if detection is ambiguous.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-file", default=None,
                   help="Log file path. Default: <out-dir>/infer_wsi_<ts>.log")
    p.add_argument("--quiet", action="store_true",
                   help="Do not echo log lines to the console (file only).")
    p.add_argument("--reuse-detections", action="store_true",
                   help="Skip inference and load cached "
                        "wsi_detections_raw.json from --out-dir, then only "
                        "re-run threshold optimisation and scoring.")
    return p.parse_args(argv)


def load_split_manifest(arg):
    """Return (val_ids, test_ids) sets from a split manifest."""
    path = Path(arg)
    if not (path.exists() and path.suffix == ".json"):
        raise ValueError(f"--slides must be an existing split_manifest.json; "
                         f"got {arg!r}")
    manifest = json.loads(path.read_text())
    slides = manifest.get("slides", {})
    val = slides.get("val")
    test = slides.get("test")
    if val is None or test is None:
        raise ValueError(
            f"{path.name} must contain both slides.val and slides.test "
            f"(found keys: {sorted(slides)}).")
    val, test = set(val), set(test)
    overlap = val & test
    if overlap:
        raise ValueError(
            f"val and test splits overlap on {len(overlap)} slide(s): "
            f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}. "
            f"Threshold selection would leak.")
    log(f"Split manifest: {len(val)} val slide(s), {len(test)} test "
        f"slide(s), no overlap.")
    return val, test


def detect_bbox_format(coco: dict, override: str = "auto") -> str:
    """Decide whether COCO `bbox` entries are 'xywh' or 'xyxy' (corners).

    If `override` is not 'auto', it is returned unchanged. Otherwise the format
    is inferred from the annotations: for xywh, bbox[2]/bbox[3] are width/height
    (independent of x/y); for xyxy they are x2/y2 and must satisfy x2 > x1 and
    y2 > y1, typically with values much larger than a mitosis is wide.

    Heuristic: a mitosis box is ~50 px. If, across many boxes, bbox[2] is almost
    always > bbox[0] AND (bbox[2]-bbox[0]) is a plausible small width while
    bbox[2] itself is large, the entries are corners. We use the rule:
    if (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]) hold for ~all boxes AND the
    median of bbox[2] is on the order of image coordinates (>> median width),
    treat as xyxy; else xywh.
    """
    if override != "auto":
        log(f"GT bbox format: {override} (forced via --bbox-format).")
        return override

    anns = coco.get("annotations", [])
    sample = [a["bbox"] for a in anns
              if a.get("category_id") == 1 and "bbox" in a]
    if not sample:
        sample = [a["bbox"] for a in anns if "bbox" in a]
    if not sample:
        log("WARNING: no bboxes to inspect; assuming xywh (COCO default).")
        return "xywh"

    arr = np.asarray(sample[:5000], dtype=float)
    b0, b1, b2, b3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Under xyxy, corners must be ordered for essentially every box.
    ordered = np.mean((b2 > b0) & (b3 > b1))
    # Under xyxy, b2/b3 are absolute coords (large); the implied width b2-b0 is
    # small (~50). Under xywh, b2/b3 ARE the width/height (small ~50) and the
    # implied "x2" = b0 (== nonsense). Compare median of b2 to median of (b2-b0).
    med_b2 = float(np.median(b2))
    med_w_if_xyxy = float(np.median(np.abs(b2 - b0)))

    looks_xyxy = (ordered > 0.99) and (med_b2 > 4.0 * max(med_w_if_xyxy, 1.0))

    fmt = "xyxy" if looks_xyxy else "xywh"
    log(f"GT bbox format auto-detected: {fmt} "
        f"(ordered={ordered:.3f}, median b2={med_b2:.1f}, "
        f"median |b2-b0|={med_w_if_xyxy:.1f}).")
    if 0.5 < ordered < 0.99:
        log("  NOTE: ordering was ambiguous; if scores look wrong, set "
            "--bbox-format explicitly.")
    return fmt


def _bbox_center(bbox, fmt: str):
    """Centre (cx, cy) of one bbox given its format."""
    a, b, c, d = bbox
    if fmt == "xyxy":
        return (a + c) * 0.5, (b + d) * 0.5
    # xywh
    return a + c * 0.5, b + d * 0.5


def load_ground_truth(ann_file, bbox_format="auto"):
    """Return (gt_points, slide_tumor, fmt) from a MIDOG++ COCO-style JSON.

    `bbox_format` is 'auto' | 'xywh' | 'xyxy'; the resolved format is returned
    so the caller can log/record it.
    """
    coco = json.loads(Path(ann_file).read_text())
    fmt = detect_bbox_format(coco, bbox_format)
    id_to_img = {im["id"]: im for im in coco["images"]}
    pts, slide_tumor = {}, {}
    for a in coco["annotations"]:
        if a.get("category_id") != 1:               # mitotic figures only
            continue
        img = id_to_img[a["image_id"]]
        slide = slide_id_from_filename(img["file_name"])
        cx, cy = _bbox_center(a["bbox"], fmt)
        pts.setdefault(slide, []).append([cx, cy])
        if "tumor_type" in img:
            slide_tumor.setdefault(slide, img["tumor_type"])
    gt_points = {s: np.asarray(p, dtype=float).reshape(-1, 2)
                 for s, p in pts.items()}
    return gt_points, slide_tumor, fmt


def detections_to_arrays(detections):
    """Convert serialised detections (slide -> [[x1,y1,x2,y2,score],...]) to
    slide -> (xyxy (N,4), scores (N,)) numpy arrays."""
    out = {}
    for slide, rows in detections.items():
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            out[slide] = (arr[:, :4].copy(), arr[:, 4].copy())
        else:
            out[slide] = (np.zeros((0, 4), np.float32),
                          np.zeros((0,), np.float32))
    return out


def arrays_to_detections(det_arrays):
    """Inverse of detections_to_arrays for JSON serialisation."""
    return {
        slide: [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                 float(s)] for b, s in zip(boxes, scores)]
        for slide, (boxes, scores) in det_arrays.items()
    }


def write_split_report(out_dir, name, det_arrays, gt_points, slides,
                       slide_tumor, det_thresh, radius, settings,
                       thr_grid=None):
    """Score `slides` at a fixed det_thresh, write per-slide CSV + metrics
    JSON, print an aggregate block, and return the metrics dict.

    If `thr_grid` is given, also compute FROC (sensitivity vs FP/slide) over
    the same threshold grid -- overall and per tumour type -- and include it in
    the metrics JSON, a dedicated FROC CSV, and the log."""
    case_results = {}
    for slide in tqdm(sorted(slides), desc=f"scoring {name}", unit="slide",
                      leave=False):
        boxes, scores = det_arrays.get(
            slide, (np.zeros((0, 4)), np.zeros((0,))))
        gt = gt_points.get(slide, np.zeros((0, 2), dtype=float))
        tp, fp, fn = score_slide(gt, boxes, scores, det_thresh, radius)
        pr, rc = precision_recall(tp, fp, fn)
        case_results[slide] = dict(
            tp=tp, fp=fp, fn=fn, f1=midog_f1(tp, fp, fn),
            precision=pr, recall=rc,
            tumor=slide_tumor.get(slide, "unknown"))

    TP = sum(c["tp"] for c in case_results.values())
    FP = sum(c["fp"] for c in case_results.values())
    FN = sum(c["fn"] for c in case_results.values())
    agg_p, agg_r = precision_recall(TP, FP, FN)
    agg_f1 = midog_f1(TP, FP, FN)

    per_tumor = {}
    for c in case_results.values():
        t = per_tumor.setdefault(c["tumor"], dict(tp=0, fp=0, fn=0))
        t["tp"] += c["tp"]; t["fp"] += c["fp"]; t["fn"] += c["fn"]

    report = out_dir / f"wsi_scores_{name}.csv"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("slide,tumor,precision,recall,f1,tp,fp,fn\n")
        for slide in sorted(case_results):
            c = case_results[slide]
            fh.write(f"{slide},{c['tumor']},{c['precision']:.4f},"
                     f"{c['recall']:.4f},{c['f1']:.4f},"
                     f"{c['tp']},{c['fp']},{c['fn']}\n")

    metrics = {
        "split": name,
        "det_thresh": det_thresh,
        "aggregates": {
            "precision": agg_p, "recall": agg_r, "f1_score": agg_f1,
            "tp": TP, "fp": FP, "fn": FN,
        },
        "per_tumor": {},
        "settings": settings,
    }
    for tumor, t in per_tumor.items():
        tp_, fp_, fn_ = t["tp"], t["fp"], t["fn"]
        tpr, trc = precision_recall(tp_, fp_, fn_)
        metrics["per_tumor"][tumor] = dict(
            precision=tpr, recall=trc, f1=midog_f1(tp_, fp_, fn_),
            tp=tp_, fp=fp_, fn=fn_)

    # ---- FROC (threshold-swept; needs the candidate grid) -----------------
    # FROC is threshold-independent in spirit -- it sweeps the threshold rather
    # than fixing it at det_thresh -- so it is computed over the same grid the
    # val sweep used. "Overall" pools all slides in this split; the per-tumour
    # curves group slides by tumour type (the only "individual" axis the
    # single-class annotations support).
    if thr_grid:
        overall_froc = _froc_block(
            det_arrays, gt_points, sorted(slides), thr_grid, radius)
        metrics["froc"] = {
            "overall": {k: v for k, v in overall_froc.items() if k != "curve"},
            "per_tumor": {},
            "curve_overall": overall_froc["curve"],
        }
        # group slides by tumour type
        tumor_to_slides = {}
        for slide in slides:
            tt = slide_tumor.get(slide, "unknown")
            tumor_to_slides.setdefault(tt, set()).add(slide)
        per_tumor_curves = {}
        for tt, tt_slides in tumor_to_slides.items():
            blk = _froc_block(det_arrays, gt_points, sorted(tt_slides),
                              thr_grid, radius)
            metrics["froc"]["per_tumor"][tt] = {
                k: v for k, v in blk.items() if k != "curve"}
            per_tumor_curves[tt] = blk["curve"]

        # FROC curve CSV: overall + per-tumour, long format for easy plotting.
        froc_csv = out_dir / f"wsi_froc_{name}.csv"
        with open(froc_csv, "w", encoding="utf-8") as fh:
            fh.write("group,threshold,fppi,sensitivity,tp,fp,fn\n")
            for p in overall_froc["curve"]:
                fh.write(f"overall,{p['threshold']:.4f},{p['fppi']:.4f},"
                         f"{p['sensitivity']:.4f},{p['tp']},{p['fp']},"
                         f"{p['fn']}\n")
            for tt in sorted(per_tumor_curves):
                for p in per_tumor_curves[tt]:
                    fh.write(f"{tt},{p['threshold']:.4f},{p['fppi']:.4f},"
                             f"{p['sensitivity']:.4f},{p['tp']},{p['fp']},"
                             f"{p['fn']}\n")

    (out_dir / f"wsi_metrics_{name}.json").write_text(
        json.dumps(metrics, indent=2))

    log(f"\n========  MIDOG-style WSI results [{name}] "
        f"(det_thresh={det_thresh:g})  ========")
    log(f"slides scored      : {len(case_results)}")
    log(f"total TP / FP / FN : {TP} / {FP} / {FN}")
    log(f"precision          : {agg_p:.4f}")
    log(f"recall             : {agg_r:.4f}")
    log(f"F1  (2TP/(2TP+FP+FN)): {agg_f1:.4f}")
    if thr_grid and "froc" in metrics:
        fo = metrics["froc"]["overall"]
        log(f"FROC score (overall): {fo['froc_score']:.4f}  "
            f"(mean sensitivity @ {{{', '.join(f'{f:g}' for f in FROC_FPPI)}}} "
            f"FP/slide)")
        sens = fo["sensitivity_at_fppi"]
        log("  sensitivity @ FP/slide: " +
            ", ".join(f"{f:g}:{sens[f'{f:g}']:.3f}" for f in FROC_FPPI))
    if len(per_tumor) > 1:
        log("-- per tumour type --")
        for tumor in sorted(per_tumor):
            t = per_tumor[tumor]
            f = midog_f1(t["tp"], t["fp"], t["fn"])
            froc_str = ""
            if thr_grid and "froc" in metrics:
                tt_froc = metrics["froc"]["per_tumor"].get(tumor)
                if tt_froc:
                    froc_str = f"  FROC={tt_froc['froc_score']:.4f}"
            log(f"  {tumor:<28s} F1={f:.4f}{froc_str}  "
                f"(TP {t['tp']}, FP {t['fp']}, FN {t['fn']})")
    log(f"per-slide CSV      : {report}")
    log(f"metrics JSON       : {out_dir / f'wsi_metrics_{name}.json'}")
    if thr_grid and "froc" in metrics:
        log(f"FROC curve CSV     : {out_dir / f'wsi_froc_{name}.csv'}")
    log("=" * 60)
    return metrics


def main(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    args = parse_args(argv)

    roi_dir = Path(args.roi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging must be set up before anything is logged.
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else \
        out_dir / f"infer_wsi_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path, args.quiet)
    log(f"Run started {ts}")
    log("Command line: " + " ".join(sys.argv))

    for label, pth in (("config", args.config),
                       ("checkpoint", args.checkpoint),
                       ("roi directory", args.roi_dir),
                       ("annotation file", args.ann_file),
                       ("split manifest", args.slides)):
        if not Path(pth).exists():
            log(f"ERROR: {label} not found: {pth}")
            return 1

    if not 0.0 <= args.overlap < 1.0:
        log("ERROR: --overlap must be in [0, 1).")
        return 1
    if not (args.thr_min < args.thr_max and args.thr_step > 0):
        log("ERROR: need --thr-min < --thr-max and --thr-step > 0.")
        return 1

    window = args.window
    stride = max(1, int(round(window * (1.0 - args.overlap))))
    log(f"Window: {window} px | overlap: {args.overlap:.0%} | "
        f"stride: {stride} px")
    log(f"Scoring: MIDOG-style, radius {args.radius:g} px "
        f"({'evalutils' if _HAVE_EVALUTILS else 'built-in KD-tree fallback'})")

    # Geometry consistency: window must equal Resize scale and backbone img_size.
    # (Read the config once here; phase 1 reads it again for the pipeline.)
    cfg_for_check = Config.fromfile(args.config)
    assert_geometry_consistent(cfg_for_check, window)

    val_slides, test_slides = load_split_manifest(args.slides)
    gt_points, slide_tumor, gt_fmt = load_ground_truth(
        args.ann_file, args.bbox_format)

    raw_path = out_dir / "wsi_detections_raw.json"

    # =====================================================================
    # PHASE 1 - threshold-free inference (or reuse cache)
    # =====================================================================
    if args.reuse_detections:
        if not raw_path.exists():
            log(f"ERROR: --reuse-detections set but {raw_path} not found.")
            return 1
        cached = json.loads(raw_path.read_text())
        det_arrays = detections_to_arrays(cached["detections"])
        log(f"Reusing cached detections for {len(det_arrays)} slide(s) "
            f"from {raw_path.name}.")
    else:
        log("Loading model...")
        model = init_detector(args.config, args.checkpoint, device=args.device)
        model.eval()
        torch.backends.cudnn.benchmark = True
        cfg = Config.fromfile(args.config)
        pipeline, bgr_to_rgb = build_inference_pipeline(cfg)
        log(f"Preprocessor bgr_to_rgb={bgr_to_rgb}; windows fed as "
            f"{'BGR' if bgr_to_rgb else 'RGB'} so the model sees RGB "
            f"(matches training).")

        # Only process slides we actually need (val + test).
        wanted = val_slides | test_slides
        roi_paths = sorted(p for p in roi_dir.iterdir()
                           if p.suffix.lower() in (".tiff", ".tif",
                                                   ".png", ".jpg"))
        roi_paths = [p for p in roi_paths if p.stem in wanted]
        if not roi_paths:
            log("ERROR: no ROI images match the val/test splits "
                "(check --roi-dir / --slides).")
            return 1
        missing = wanted - {p.stem for p in roi_paths}
        if missing:
            log(f"WARNING: {len(missing)} manifest slide(s) have no ROI "
                f"image and will score as all-FN: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        log(f"Processing {len(roi_paths)} whole-slide image(s) "
            f"(score floor {args.score_floor:g}).")

        det_arrays = {}
        for roi_path in tqdm(roi_paths, desc="WSI inference", unit="slide"):
            with Image.open(roi_path) as im:
                roi = np.asarray(im.convert("RGB"))
            boxes, scores = infer_slide(
                model, pipeline, bgr_to_rgb, roi, window, stride,
                args.nms_iou, args.score_floor, desc=roi_path.stem)
            det_arrays[roi_path.stem] = (boxes, scores)

        raw_path.write_text(json.dumps(dict(
            window=window, overlap=args.overlap, stride=stride,
            score_floor=args.score_floor, nms_iou=args.nms_iou,
            radius=args.radius, gt_bbox_format=gt_fmt,
            detections=arrays_to_detections(det_arrays),
        ), indent=2))
        log(f"\nRaw detections cached: {raw_path}")

    # =====================================================================
    # PHASE 2 - optimise det_thresh on val, then report test
    # =====================================================================
    n_steps = int(round((args.thr_max - args.thr_min) / args.thr_step)) + 1
    thr_grid = [round(args.thr_min + i * args.thr_step, 6)
                for i in range(n_steps)]
    # never select below the inference floor (those detections were discarded)
    thr_grid = [t for t in thr_grid if t >= args.score_floor]
    if not thr_grid:
        log("ERROR: threshold grid is empty after applying --score-floor.")
        return 1

    best_thr, sweep = optimise_threshold(
        det_arrays, gt_points, val_slides, thr_grid, args.radius)

    val_best = next(r for r in sweep if r["threshold"] == best_thr)
    log(f"\n---- validation threshold sweep "
        f"({thr_grid[0]:g}..{thr_grid[-1]:g} step {args.thr_step:g}) ----")
    log(f"best det_thresh    : {best_thr:g}")
    log(f"val F1 @ best      : {val_best['f1']:.4f} "
        f"(P {val_best['precision']:.4f}, R {val_best['recall']:.4f}, "
        f"TP {val_best['tp']}, FP {val_best['fp']}, FN {val_best['fn']})")

    # persist the full sweep for plotting / auditing
    (out_dir / "wsi_val_threshold_sweep.json").write_text(json.dumps(dict(
        radius=args.radius, score_floor=args.score_floor,
        grid=dict(min=args.thr_min, max=args.thr_max, step=args.thr_step),
        best_threshold=best_thr, sweep=sweep,
    ), indent=2))
    with open(out_dir / "wsi_val_threshold_sweep.csv", "w",
              encoding="utf-8") as fh:
        fh.write("threshold,precision,recall,f1,tp,fp,fn\n")
        for r in sweep:
            fh.write(f"{r['threshold']:.4f},{r['precision']:.4f},"
                     f"{r['recall']:.4f},{r['f1']:.4f},"
                     f"{r['tp']},{r['fp']},{r['fn']}\n")

    settings = {
        "radius": args.radius, "det_thresh": best_thr,
        "score_floor": args.score_floor, "nms_iou": args.nms_iou,
        "window": window, "overlap": args.overlap,
        "gt_bbox_format": gt_fmt,
        "scorer": "evalutils" if _HAVE_EVALUTILS else "fallback",
        "threshold_selected_on": "val",
    }

    # Report val at the chosen threshold (sanity) and test (the headline).
    write_split_report(out_dir, "val", det_arrays, gt_points, val_slides,
                       slide_tumor, best_thr, args.radius, settings,
                       thr_grid=thr_grid)
    write_split_report(out_dir, "test", det_arrays, gt_points, test_slides,
                       slide_tumor, best_thr, args.radius, settings,
                       thr_grid=thr_grid)

    log(f"\nThreshold {best_thr:g} was selected to maximise pooled MIDOG F1 "
        f"on the validation slides, then applied unchanged to the test "
        f"slides. Test numbers above are the ones to report.")
    log(f"Full run log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
