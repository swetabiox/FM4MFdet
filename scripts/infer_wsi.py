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

Image.MAX_IMAGE_PIXELS = None 



LOGGER = logging.getLogger("infer_wsi")


def setup_logging(log_path: Path, quiet: bool) -> None:
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
        sh.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(sh)

    LOGGER.info("Logging to %s", log_path)


def log(msg: str = "") -> None:
    """Drop-in for print() that routes through LOGGER (file + console)."""
    LOGGER.info(msg)



_TILE_SUFFIX = re.compile(
    r"(?:[_-](?:x\d+[_-]y\d+|tile[_-]?\d+|patch[_-]?\d+|\d+[_-]\d+|\d+))+$",
    re.IGNORECASE,
)


def slide_id_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stripped = _TILE_SUFFIX.sub("", stem)
    return stripped if stripped else stem



def window_origins(extent: int, window: int, stride: int):
    if extent <= window:
        return [0]
    origins = list(range(0, extent - window + 1, stride))
    last = extent - window
    if origins[-1] != last:
        origins.append(last)
    return origins




def _resize_scale_from_pipeline(cfg):
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "Resize":
            scale = t.get("scale")
            if isinstance(scale, (tuple, list)) and len(scale) == 2:
                return tuple(int(v) for v in scale)
            if isinstance(scale, int):
                return (scale, scale)
    return None


def _backbone_img_size(cfg):
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        bb = model_cfg.get("backbone", {})
        if isinstance(bb, dict):
            v = bb.get("img_size")
            if v is not None:
                return int(v)
    return None


def assert_geometry_consistent(cfg, window: int) -> None:
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




def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
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




def _centers(xyxy: np.ndarray) -> np.ndarray:
    """Box centers from [x1,y1,x2,y2] rows -> (N,2) array of (cx,cy)."""
    if xyxy.size == 0:
        return np.zeros((0, 2), dtype=float)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) * 0.5,
                     (xyxy[:, 1] + xyxy[:, 3]) * 0.5], axis=1)


try:
    from evalutils.scorers import score_detection as _evalutils_score
    _HAVE_EVALUTILS = True
except Exception:                                 
    _evalutils_score = None
    _HAVE_EVALUTILS = False


def _score_detection_fallback(gt_points: np.ndarray,
                              pred_points: np.ndarray,
                              radius: float):
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


FROC_FPPI = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def froc_curve(det_arrays, gt_points, slides, thr_grid, radius):
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
    pts.sort(key=lambda d: d["fppi"])
    return pts


def froc_score(curve, fppi_points=FROC_FPPI):
    per_point = {}
    if not curve:
        for f in fppi_points:
            per_point[f] = 0.0
        return 0.0, per_point

    max_fppi = curve[-1]["fppi"]
    max_sens_overall = max(p["sensitivity"] for p in curve)

    for f in fppi_points:
        if f >= max_fppi:
            per_point[f] = max_sens_overall
            continue
        cands = [p["sensitivity"] for p in curve if p["fppi"] <= f]
        per_point[f] = max(cands) if cands else 0.0

    score = float(np.mean([per_point[f] for f in fppi_points]))
    return score, per_point


def _froc_block(det_arrays, gt_points, slides, thr_grid, radius):
    curve = froc_curve(det_arrays, gt_points, slides, thr_grid, radius)
    score, per_point = froc_score(curve)
    return dict(
        froc_score=score,
        fppi_points=list(FROC_FPPI),
        sensitivity_at_fppi={f"{f:g}": per_point[f] for f in FROC_FPPI},
        curve=curve,
    )


@torch.no_grad()
def detect_window(model, pipeline, window_rgb: np.ndarray, bgr_to_rgb: bool):
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
    steps = []
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get('type') in ('LoadImageFromFile', 'LoadAnnotations'):
            continue
        steps.append(t)
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


def pooled_f1_at(det_arrays, gt_points, slides, det_thresh, radius):
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
    sweep = []
    best = None
    for thr in tqdm(thr_grid, desc="val threshold sweep", unit="thr",
                    leave=False):
        f1, p, r, TP, FP, FN = pooled_f1_at(
            det_arrays, gt_points, val_slides, thr, radius)
        row = dict(threshold=float(thr), f1=f1, precision=p, recall=r,
                   tp=TP, fp=FP, fn=FN)
        sweep.append(row)
        key = (f1, thr)
        if best is None or key > best[0]:
            best = (key, thr)
    return float(best[1]), sweep


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

    ordered = np.mean((b2 > b0) & (b3 > b1))
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
    coco = json.loads(Path(ann_file).read_text())
    fmt = detect_bbox_format(coco, bbox_format)
    id_to_img = {im["id"]: im for im in coco["images"]}
    pts, slide_tumor = {}, {}
    for a in coco["annotations"]:
        if a.get("category_id") != 1:              
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
    return {
        slide: [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                 float(s)] for b, s in zip(boxes, scores)]
        for slide, (boxes, scores) in det_arrays.items()
    }


def write_split_report(out_dir, name, det_arrays, gt_points, slides,
                       slide_tumor, det_thresh, radius, settings,
                       thr_grid=None):
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

    if thr_grid:
        overall_froc = _froc_block(
            det_arrays, gt_points, sorted(slides), thr_grid, radius)
        metrics["froc"] = {
            "overall": {k: v for k, v in overall_froc.items() if k != "curve"},
            "per_tumor": {},
            "curve_overall": overall_froc["curve"],
        }
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
    cfg_for_check = Config.fromfile(args.config)
    assert_geometry_consistent(cfg_for_check, window)

    val_slides, test_slides = load_split_manifest(args.slides)
    gt_points, slide_tumor, gt_fmt = load_ground_truth(
        args.ann_file, args.bbox_format)

    raw_path = out_dir / "wsi_detections_raw.json"

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


    n_steps = int(round((args.thr_max - args.thr_min) / args.thr_step)) + 1
    thr_grid = [round(args.thr_min + i * args.thr_step, 6)
                for i in range(n_steps)]
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
